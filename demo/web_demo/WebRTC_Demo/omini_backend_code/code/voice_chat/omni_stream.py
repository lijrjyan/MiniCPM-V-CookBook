import asyncio
from collections import deque
from datetime import datetime
import os
import uuid
import numpy as np
from scipy.signal import resample_poly
from services.inference_service_manager import InferenceService, get_service_manager
from voice_chat.entity.token import LoginRequest
from voice_chat.entity.session import SharedSessionState
from voice_chat.vad import vad_utils
import time

from enhanced_logging_config import get_enhanced_logger, set_request_trace
from voice_chat.model_call import MiniCpmModel
from common.enums.model_type import ModelType
from concurrent.futures import ThreadPoolExecutor
from config.settings import get_voice_chat_settings
from voice_chat.stream_resample import StreamResampler

# 获取日志器
logger = get_enhanced_logger('voice_chat')

# VAD 检测专用线程池（独立于全局线程池，避免被其他长时间任务阻塞）
# 增加worker数量以支持更多并发用户（每个用户并发VAD检测）
_vad_thread_pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="VAD")

# Latency patch: audio goes to the inference service in PREFILL_CHUNK_MS pieces (upstream: 1000), and duplex keeps
# one streaming_generate open at a time (reopened when it ends) instead of starting one per chunk.
PREFILL_CHUNK_MS = int(os.getenv("OMNI_PREFILL_CHUNK_MS", "80"))
DUPLEX_KEEP_OPEN = os.getenv("OMNI_DUPLEX_KEEP_OPEN", "1") == "1"


class OmniStream:
    def __init__(self, inference_service: InferenceService, request: LoginRequest, 
    audio_input_queue: asyncio.Queue,
    audio_output_queue: asyncio.Queue,
    text_output_queue: asyncio.Queue,
    stop_event:asyncio.Event,
    model_cpm:MiniCpmModel,
    shared_state: SharedSessionState):
        self.audio_input_queue = audio_input_queue
        self.audio_output_queue = audio_output_queue
        self.text_output_queue = text_output_queue
        self.inference_service = inference_service
        self.model_cpm = model_cpm
        self.stop_event = stop_event
        self.active_tasks = set()
        self.session_id = request.sessionId
        self.user_id = request.userId
        self.session_type = request.sessionType
        # vad
        self.vad_options = vad_utils.VadOptions()
        self.vad_stream_started = False
        self.vad_time = time.time()
        self.dur_vad_time = request.durVadTime
        self.dur_vad_threshold = request.durVadThreshold
        self.vad_race = request.vadRace
        # 准备抢跑的标识，第二次小于0.1才返回可以抢跑
        self.vad_race_prepare = False
        self.vad_race_flag = asyncio.Event()
        self.vad_race_audio_queue = asyncio.Queue()
        self.vad_race_text_queue = asyncio.Queue()
        self.vad_race_task = None  # 跟踪当前抢跑任务

        # ordered prefill sender and the stateful 48 kHz -> 16 kHz input resampler
        self.prefill_queue = asyncio.Queue()
        self.prefill_task = None
        self.input_resampler = StreamResampler(48000, 16000)
        self.generate_loop_task = None

        # 音频配置
        self.WEBRTC_SAMPLE_RATE = 48000
        self.WEBRTC_CHUNK_SIZE = self.WEBRTC_SAMPLE_RATE//10
        self.BUFFER_SIZE = self.WEBRTC_SAMPLE_RATE
        
        # 协程间共享的会话状态
        self.shared_state = shared_state
        
        # 从配置文件中读取语音打断相关配置
        voice_chat_config = get_voice_chat_settings()
        self.enable_voice_interruption = voice_chat_config.enable_voice_interruption
        self.voice_interruption_threshold = voice_chat_config.voice_interruption_threshold

    async def _collect_audio_data(self) -> np.ndarray:
        """
        收集音频输入队列中的所有可用数据
        
        Returns:
            combined_audio_data: np.ndarray
        """
        collected_data = []
        # 使用 get_nowait() 避免阻塞，循环收集所有可用数据
        while True:
            try:
                audio_data = self.audio_input_queue.get_nowait()
                if audio_data is None:
                    break
                # 处理带时间戳的音频数据
                if isinstance(audio_data, tuple) and len(audio_data) == 3:
                    audio_array, _, _ = audio_data
                    collected_data.append(audio_array)
                    
            except asyncio.QueueEmpty:
                # 队列为空，正常退出循环
                break
            except Exception as e:
                logger.error(f"收集音频数据错误: {str(e)}")
                break
        
        return collected_data

    async def _process_audio_batch(self, audio_data_buffer, buffer_duration, target_duration):
        """
        预填音频批次数据
        
        Returns:
            tuple: (updated_audio_data_buffer, updated_buffer_duration)
        """
        # 计算需要保留的数据量
        logger.debug(f"process_audio_batch buffer_duration: {buffer_duration}, target_duration: {target_duration}")
        remaining_duration = buffer_duration % target_duration
        remaining_samples = int((remaining_duration / 1000) * self.WEBRTC_SAMPLE_RATE)

        # 合并所有缓冲的音频数据
        full_audio_data = np.concatenate(audio_data_buffer)

        try:
            # 分割数据，保留超过1000ms的部分
            if remaining_samples > 0:
                # 发送前1000ms的数据
                await self.model_prefill(full_audio_data[:-remaining_samples])
                # 保留剩余数据
                updated_audio_data_buffer = [full_audio_data[-remaining_samples:]]
                updated_buffer_duration = remaining_duration
            else:
                # 如果刚好是1000ms，发送全部数据
                await self.model_prefill(full_audio_data)
                updated_audio_data_buffer = []
                updated_buffer_duration = 0
            return updated_audio_data_buffer, updated_buffer_duration
        finally:
            # 清理内存
            del full_audio_data

    async def _handle_model_generate(self) -> None:
        """
        处理未检测到语音活动的情况(SIMPLEX模式)
        """
        if self.model_cpm.model_type == ModelType.SIMPLEX and not self.model_cpm.play_end_event.is_set():
            logger.info("模型和前端正在输出,忽略generate")
            return
        if self.model_cpm.model_type == ModelType.SIMPLEX:
            await self.text_output_queue.put("<state><vad_end>")
            await self.text_output_queue.put(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} - <state><vad_end>")
            self.vad_stream_started = False
        output_resampler = None
        try:
            round_id = await self.shared_state.get_round()
            generator = self.model_cpm.streaming_generate(
                session_id=self.session_id,
                keep_open=DUPLEX_KEEP_OPEN and self.model_cpm.model_type == ModelType.DUPLEX,
            )
            # 每次生成后进行续命服务锁定
            service_manager = await get_service_manager()
            await service_manager.renew_service_lock(self.inference_service.locked_by, self.inference_service.service_id)
            await self.text_output_queue.put("<state><generate_start>")
            
            async for chunk in generator:
                logger.info(f"收到流式数据: {chunk}")
                
                # 检查是否是结束标志
                if chunk.get('type') == 'done':
                    break
                
                # 解析流式数据中的音频内容
                chunk_data = chunk.get('chunk_data')
                if chunk_data:
                    # 获取音频数据和采样率
                    wav_data = chunk_data.get('wav')
                    tts_sample_rate = chunk_data.get('sample_rate', 24000)

                    audio_data = None
                    if wav_data is not None:
                        # 重采样到 WebRTC 采样率 (stateful across this stream's chunks, off the event loop)
                        if output_resampler is None:
                            output_resampler = StreamResampler(tts_sample_rate, self.WEBRTC_SAMPLE_RATE)
                        audio_data = await asyncio.to_thread(output_resampler.process, wav_data)
                        # 将音频数据放入队列 (published at once: no first-reply hold)
                        await self.audio_output_queue.put(audio_data)
                    # 处理文本内容
                    text_content = chunk_data.get('text')
                    if text_content is not None:
                        await self.text_output_queue.put(text_content)
        except Exception as e:
            logger.error(f"模型生成错误: {str(e)}")
        finally:
            await self.text_output_queue.put("<state><generate_end>")

    def _clear_audio_queues(self) -> None:
        """
        清理音频队列和缓冲区
        """
        # 清空音频输入队列
        while not self.audio_input_queue.empty():
            try:
                self.audio_input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    async def _async_stream_detail(self, audio_buffer):
        """
        异步版本的流式处理主循环
        """
        # 设置日志追踪上下文，用于关联同一会话的所有日志
        set_request_trace(
            request_id=self.session_id[:8] if self.session_id else None,
            session_id=self.session_id
        )
        logger.info(f"OmniStream 开始处理，session_id: {self.session_id}, user_id: {self.user_id}")
        
        # 任务管理集合
        # 使用实例变量来管理任务，确保回调函数能正确访问
        audio_data_buffer = []
        buffer_duration = 0
        target_duration = PREFILL_CHUNK_MS  # 目标缓冲区时长（毫秒）
        duplex = self.model_cpm.model_type == ModelType.DUPLEX
        while not self.stop_event.is_set():
            try:
                start_time = time.time()
                
                # 1. 收集音频数据 (duplex: wake on the first frame instead of polling every 100 ms)
                if duplex:
                    try:
                        first = await asyncio.wait_for(self.audio_input_queue.get(), timeout=0.2)
                    except asyncio.TimeoutError:
                        continue
                    collected_data = ([first[0]] if isinstance(first, tuple) and len(first) == 3 else []) + await self._collect_audio_data()
                else:
                    collected_data = await self._collect_audio_data()
                if collected_data:
                    # 2. 处理音频缓冲区 (duplex needs no VAD window)
                    combined_data = np.concatenate(collected_data)
                    if not duplex:
                        audio_buffer.extend(combined_data)
                    if duplex or len(audio_buffer) >= self.BUFFER_SIZE:
                        if self.model_cpm.model_type == ModelType.SIMPLEX:
                            # SIMPLEX模式：需要VAD检测
                            # 使用专用 VAD 线程池执行检测，避免被其他长时间任务阻塞
                            loop = asyncio.get_event_loop()
                            full_vad_result, tail_vad_result, dur_vad_full = await loop.run_in_executor(
                                _vad_thread_pool, self.vad_dual_detection, audio_buffer)
                            # 如果检测到有语音活动,但是模型正在输出,强制打断模型
                            # 从配置文件中读取配置来判断是否需要语音打断
                            if (self.enable_voice_interruption and 
                                not self.model_cpm.play_end_event.is_set() and 
                                full_vad_result and 
                                dur_vad_full > self.voice_interruption_threshold):
                                asyncio.create_task(self.model_cpm.streaming_break(session_id=self.session_id, text=f"说话打断"))
                            # SIMPLEX模式：需要VAD检测
                            if full_vad_result:
                                # 计算当前数据块的时长（毫秒）
                                chunk_duration = (len(combined_data) / self.WEBRTC_SAMPLE_RATE) * 1000
                                buffer_duration += chunk_duration
                                audio_data_buffer.append(combined_data)

                                # 当缓冲区达到目标时长时，处理数据
                                if buffer_duration >= target_duration:
                                    audio_data_buffer, buffer_duration = await self._process_audio_batch(audio_data_buffer, buffer_duration, target_duration)
                                if self.vad_race:
                                    # 抢跑模式
                                    if not tail_vad_result:
                                        if not self.vad_race_flag.is_set() and self.vad_race_task is None:
                                            # 执行vad抢跑的逻辑
                                            self.vad_race_flag.set()
                                            self.vad_race_task = asyncio.create_task(self._handle_model_generate())
                                        else:
                                            logger.info("抢跑任务已存在，跳过创建新任务")
                                    else:
                                        if self.vad_race_flag.is_set():
                                            # 抢跑失败,停止抢跑
                                            self.vad_race_flag.clear()
                                            self.vad_race_task = None
                                            asyncio.create_task(self.stop_vad_race_encode())
                            else:
                                # 无语音活动，生成响应
                                if self.vad_race_flag.is_set():
                                    logger.info(f"抢跑成功,释放queue的数据")
                                    # 抢跑成功释放queue的数据
                                    while self.vad_race_task is not None:
                                        try:
                                            audio_data = self.vad_race_audio_queue.get_nowait()
                                            if audio_data:
                                                await self.audio_input_queue.put(audio_data)
                                        except asyncio.QueueEmpty:
                                            break
                                        try:
                                            text_data = self.vad_race_text_queue.get_nowait()
                                            if text_data:
                                                await self.text_output_queue.put(text_data)
                                        except asyncio.QueueEmpty:
                                            break
                                    self.vad_race_flag.clear()
                                else:
                                    # 判断剩下的audio_data_buffer是否大于0.1s,如果大于0.1s,则使用静音拼接生成1s的音频发送给模型
                                    if buffer_duration > 50 and len(audio_data_buffer) > 0:
                                        # 发送尾巴音频数据
                                        existing_audio = np.concatenate(audio_data_buffer)
                                        await self.model_prefill(existing_audio, last_chunk=True)
                                    asyncio.create_task(self._handle_model_generate())
                                # 单工输出之后清理之前的缓冲区数据
                                self.input_resampler.reset()
                                audio_buffer.clear()
                                buffer_duration = 0
                                audio_data_buffer = []
                                self._clear_audio_queues()
                        elif self.model_cpm.model_type == ModelType.DUPLEX:
                            # DUPLEX模式：不需要VAD检测，直接处理音频数据
                            # 计算当前数据块的时长（毫秒）
                            chunk_duration = (len(combined_data) / self.WEBRTC_SAMPLE_RATE) * 1000
                            buffer_duration += chunk_duration
                            audio_data_buffer.append(combined_data)

                            # 当缓冲区达到目标时长时，处理数据
                            if buffer_duration >= target_duration:
                                audio_data_buffer, buffer_duration = await self._process_audio_batch(audio_data_buffer, buffer_duration, target_duration)
                                # 双工模式：尝试获取用户输入的文本数据，模型返回的数据
                                if not DUPLEX_KEEP_OPEN:
                                    asyncio.create_task(self._handle_model_generate())
                                elif self.generate_loop_task is None:
                                    self.generate_loop_task = asyncio.create_task(self._duplex_generate_loop())
                else:
                    # 没有音频数据时，短暂休眠
                    logger.debug("未收集到音频数据，继续等待...")
                    await asyncio.sleep(0.1)

                # 4. 控制循环频率（使用异步 sleep，不阻塞事件循环）
                elapsed = time.time() - start_time
                sleep_time = max(0, 0.1 - elapsed)
                if sleep_time > 0 and not duplex:
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"音频处理错误: {str(e)}")
                continue
        for task in (self.generate_loop_task, self.prefill_task):
            if task is not None:
                task.cancel()
        await self.model_cpm.streaming_stop(session_id=self.session_id)
        logger.info(f"omniStream结束")

    async def _duplex_generate_loop(self):
        """Duplex: keep exactly one streaming_generate open; reopen it as soon as the service ends it."""
        while not self.stop_event.is_set():
            started = time.monotonic()
            await self._handle_model_generate()
            if time.monotonic() - started < 0.1:
                await asyncio.sleep(0.1)  # the service refused or ended at once: do not spin

    async def model_prefill(self, audio_data: np.ndarray, last_chunk: bool = False):
        """
        模型预填: resampled to 16 kHz here (stateful) and sent in order by one sender task, so short chunks
        cannot overtake each other on parallel HTTP connections.
        """
        self.prefill_queue.put_nowait((self.input_resampler.process(audio_data), last_chunk))
        if self.prefill_task is None:
            self.prefill_task = asyncio.create_task(self._prefill_sender())

    async def _prefill_sender(self):
        while True:
            audio_data, last_chunk = await self.prefill_queue.get()
            # after a stall, send everything queued (up to a last_chunk marker) as one request
            while not last_chunk and not self.prefill_queue.empty():
                more, last_chunk = self.prefill_queue.get_nowait()
                audio_data = np.concatenate([audio_data, more])
            try:
                await self.model_cpm.model_prefill(self.session_id, audio_data=audio_data, last_chunk=last_chunk, audio_sample_rate=16000)
            except Exception as e:
                logger.error(f"调用 model_prefill 失败: {e}")



    def vad(self, audio_buffer: deque):
        """
        音频vad检测
        """
        start_time = time.time()
        
        # 数据转换阶段
        buffer_array = np.array(list(audio_buffer))
        buffer_bytes = buffer_array.tobytes()
        # VAD 处理阶段（最耗时）
        dur_vad, _, _ = vad_utils.run_vad(
            buffer_bytes, self.WEBRTC_SAMPLE_RATE, self.vad_options)
        
        # 总耗时
        total_time = time.time() - start_time
        # 性能监控日志
        # 性能警告
        if total_time > 0.1:  # 超过100ms
            logger.info(f"VAD处理耗时过长: {total_time*1000:.2f}ms")
        
        if dur_vad > 0.4:
            if self.vad_stream_started == False:
                self.vad_time = time.time()
                self.vad_stream_started = True
        elif dur_vad < 0.2:
            if self.vad_stream_started:
                self.vad_stream_started = False
                if (time.time() - self.vad_time >= 0.3):
                    return False
        return True

    def vad_dual_detection(self, audio_buffer):
        """
        双重VAD检测方法
        1. 对1秒音频进行完整VAD检测
        2. 对最后0.2秒音频进行额外VAD检测
        
        Args:
            audio_buffer: 1秒音频缓冲区
            
        Returns:
            tuple: (full_vad_result, tail_vad_result)
                - full_vad_result: 1秒音频的VAD检测结果 (bool)
                - tail_vad_result: 最后0.2秒音频的VAD检测结果 (bool)
        """
        # 在线程池中执行时，contextvars 不会自动传播，需要手动设置追踪上下文
        set_request_trace(
            request_id=self.session_id[:8] if self.session_id else None,
            session_id=self.session_id
        )
        start_time = time.time()
        
        # 数据转换阶段
        buffer_array = np.array(list(audio_buffer))
        buffer_bytes = buffer_array.tobytes()
        
        # 1. 完整1秒音频的VAD检测
        dur_vad_full, _, _ = vad_utils.run_vad(
            buffer_bytes, self.WEBRTC_SAMPLE_RATE, self.vad_options)
        
        # 2. 提取最后0.秒音频
        # 计算最后0.秒对应的样本数
        tail_samples = int(self.dur_vad_time * self.WEBRTC_SAMPLE_RATE)  # 0.2秒 * 48000Hz = 9600个样本
        tail_audio = buffer_array[-tail_samples:]  # 取最后9600个样本
        # 再补充剩下的空白音频
        silence_samples = int((1- self.dur_vad_time) * self.WEBRTC_SAMPLE_RATE)  # 0.8秒 * 48000Hz = 38400个样本
        silence_audio = np.zeros(silence_samples, dtype=np.int16)  # 静音数据
        tail_audio = np.concatenate([tail_audio, silence_audio])
        tail_bytes = tail_audio.tobytes()
        
        # 3. 最后0.2秒音频的VAD检测
        dur_vad_tail, _, _ = vad_utils.run_vad(
            tail_bytes, self.WEBRTC_SAMPLE_RATE, self.vad_options)
        
        # 总耗时
        total_time = time.time() - start_time
        
        # 性能监控
        if total_time > 0.1:  # 超过200ms
            logger.info(f"双重VAD处理耗时过长: {total_time*1000:.2f}ms")
        
        # 4. 判断逻辑
        # 完整1秒音频的VAD检测结果
        full_vad_result = True
        if dur_vad_full > 0.4:
            if self.vad_stream_started == False:
                self.vad_time = time.time()
                self.vad_stream_started = True
        elif dur_vad_full < 0.1:
            if self.vad_stream_started:
                self.vad_stream_started = False
                if (time.time() - self.vad_time >= 0.3):
                    full_vad_result = False
        logger.info(f"dur_vad_full: {dur_vad_full}, dur_vad_tail: {dur_vad_tail}")
        # 最后0.2秒音频的VAD检测结果
        tail_vad_result = True
        if self.vad_stream_started:
            #这里连续两次小于0.1才返回可以抢跑,减小误判几率
            if dur_vad_tail < self.dur_vad_threshold:
                if self.vad_race_prepare:
                    tail_vad_result = False
                    self.vad_race_prepare = False
                else:
                    self.vad_race_prepare = True
        
        return full_vad_result, tail_vad_result, dur_vad_full
    

    async def vad_race_decode(self):
        """
        抢跑模型预解码
        """
        try:
            logger.info(f"抢跑模型预解码启动")
            await self.text_output_queue.put(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} - <state><vad_end>")
            # if self.vad_race_flag.is_set():
            #     logger.info(f"抢跑模型预解码已启动,忽略本次抢跑")
            #     return
            generator = self.model_cpm.streaming_generate(session_id=self.session_id)
            await self.text_output_queue.put("<state><generate_start>")
            async for chunk in generator:
                logger.info(f"抢跑收到流式数据: {chunk}")
                if not self.vad_race_flag.is_set():
                    logger.info(f"抢跑失败,丢失抢跑数据")
                    break
                # 解析流式数据中的文本内容
                text_content = chunk['content']
                if text_content:
                    # 将文本内容放入队列
                    await self.text_output_queue.put(text_content)
                
                if chunk.get('type') == 'done':
                    break
            await self.text_output_queue.put("<state><generate_end>")
        except Exception as e:
            logger.error(f"模型生成错误: {str(e)}")
        finally:
            self.vad_race_task = None  # 清理任务引用
            logger.info(f"抢跑模型预解码结束")

    async def stop_vad_race_encode(self):
        """
        停止抢跑模型预解码
        """ 
        logger.info(f"抢跑失败,停止模型预解码")
        # todo 通知模型侧抢跑失败
        # 清空抢跑队列中的数据
        while not self.vad_race_audio_queue.empty():
            try:
                self.vad_race_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self.vad_race_text_queue.empty():
            try:
                self.vad_race_text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def process_audio_chunk(self, audio_data, input_rate, output_rate, chunk_size):
        try:
            resampled_data = resample_poly(
                audio_data, output_rate, input_rate, padtype='line')
            resampled_data = np.clip(
                resampled_data, -32768, 32767).astype(np.int16)
            total_chunks = len(resampled_data) // chunk_size
            chunks = []
            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunk = resampled_data[start_idx:end_idx]
                chunks.append(chunk)
            return chunks
        except Exception as e:
            print(f"音频处理错误: {str(e)}")
            return []