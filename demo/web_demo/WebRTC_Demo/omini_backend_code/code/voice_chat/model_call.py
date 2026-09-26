

import asyncio
import base64
from datetime import datetime
import json
import numpy as np
from typing import Dict, Any, Optional, Union, Generator
from common.enums.model_type import ModelType
from common.utils.httpUtil import get_async_http_util, HTTPUtilError
from enhanced_logging_config import get_enhanced_logger
from services.inference_service_manager import InferenceService, get_service_manager
from voice_chat.entity.session import SharedSessionState
from voice_chat.entity.token import LoginRequest

# 获取日志器
logger = get_enhanced_logger('model_call')
    
class MiniCpmModel ():
   WEBRTC_SAMPLE_RATE = 48000
   
   def __init__(self, inference_service: InferenceService, request: LoginRequest, 
   text_output_queue: asyncio.Queue, audio_output_queue: asyncio.Queue, 
   first_tts: asyncio.Event, shared_state: SharedSessionState, model_generating_flag: asyncio.Event):
     self.request = request
     self.model_type = ModelType.get_model_name(model_type=request.modelType)
     self.api_base_url = f"http://{inference_service.ip}:{inference_service.model_port}"
     self.break_url = f"http://{inference_service.ip}:{inference_service.model_port+1}"
     # 使用全局单例 HTTP 客户端，共享连接池
     self.http_util = get_async_http_util(max_retries=3)
     # 模型是否正在输出
     self.text_output_queue = text_output_queue
     self.audio_output_queue = audio_output_queue
     self.play_end_event = asyncio.Event()
     self.first_tts = first_tts
     self.inference_service = inference_service
     # 初始化时设置为已结束
     self.play_end_event.set()
     self.shared_state = shared_state
     self.model_generating_flag = model_generating_flag

   async def model_init(self):
        try:
            data = {
                "highRefresh": self.request.highRefresh,
                "highImage": self.request.highImage,
                "timbreId": self.request.timbreId,
                "timbreBase64": self.request.base64String,
                **self.request.modelConfig.model_dump()
            }
            if self.request.language is not None:
                data["language"] = self.request.language
            response = await self.http_util.post(
                url=f"{self.api_base_url}/omni/init_sys_prompt",
                data=json.dumps(data),
                headers={'Content-Type': 'application/json'}
            )
            logger.info(f"模型初始化请求req:{data}, 返回结果: {response}")
            if response['success']:
                logger.info(f"模型初始化成功: {response['status_code']}")
                await self.text_output_queue.put("<state><model_init_success>")
                return response['data']
            else:
                service_manager = await get_service_manager()
                await service_manager.release_service_lock(self.inference_service.service_id, self.request.userId)
                logger.info(f"模型初始化失败, 释放服务锁定: {self.inference_service.service_id}")
                await self.text_output_queue.put(f"<state><model_init_failed>")
                raise HTTPUtilError(f"API请求失败: HTTP {response['status_code']}")
        except Exception as e:
            service_manager = await get_service_manager()
            await service_manager.release_service_lock(self.inference_service.service_id, self.request.userId)
            logger.info(f"模型初始化失败, 释放服务锁定: {self.inference_service.service_id}")
            await self.text_output_queue.put(f"<state><model_init_failed>")
            logger.error(f"模型初始化失败: {str(e)}")
            raise HTTPUtilError(f"模型初始化失败: {str(e)}")


   async def model_prefill(self, session_id: str, audio_data: Optional[np.ndarray] = None,
        image_data: Optional[Union[np.ndarray, bytes]] = None, last_chunk: bool = False,
        audio_sample_rate: int = WEBRTC_SAMPLE_RATE) -> Dict[str, Any]:
        roundId = await self.shared_state.get_round()
        image_audio_id = await self.shared_state.get_current_image_audio_id()
        # 模型如果是单工并且正在输出，则直接返回
        if self.model_type == ModelType.SIMPLEX and not self.play_end_event.is_set():
            logger.info(f"模型和前端正在输出,忽略prefill")
            return {"success": True, "data": {"message": "prefill success"}}        
        # 编码为 base64
        audio_content = None
        if audio_data is not None and len(audio_data) > 0:
            audio_content = self._encode_audio_to_base64(audio_data, input_sample_rate=audio_sample_rate, output_sample_rate=16000)
        image_content = None
        if image_data is not None:
            image_content = self._encode_image_to_base64(image_data, image_format="jpeg")
        return await self.streaming_prefill(session_id=session_id, audio_data=audio_content, image_data=image_content, 
        roundId=roundId, image_audio_id=image_audio_id, last_chunk=last_chunk)

   async def streaming_prefill(
         self,
         session_id: str,
         audio_data: str,
         image_data: str,
         roundId: int,
         image_audio_id: int,
         last_chunk: bool = False) -> Dict[str, Any]:
       """
       Omni流式输入接口
       """
       try:
           # 构建请求数据
           request_data = {
               "session_id": session_id,
               "audio": audio_data,
               "image": image_data,
               "image_audio_id": image_audio_id,
               "round": roundId,
               "last_chunk": last_chunk
           }
           
           # 构建API URL
           api_url = f"{self.api_base_url}/omni/streaming_prefill"
                      
           # 发送POST请求
           response = await self.http_util.post(
               url=api_url,
               json_data=request_data,
               headers={'Content-Type': 'application/json'}
           )
           logger.debug(f"Omni prefill请求返回结果: {response}")  # latency patch: per-chunk lines at debug (12.5 chunks/s)
           if response['success']:
               logger.debug(f"Omni prefill请求成功: {response['status_code']}")
               return response['data']
           else:
               #logger.error(f"Omni prefill请求失败: {response['status_code']} - {response.get('data', 'Unknown error')}, request_data: {request_data}")
               raise HTTPUtilError(f"API请求失败: HTTP {response['status_code']}")
               
       except Exception as e:
           logger.error(f"Omni prefill请求异常: {str(e)}")
           raise HTTPUtilError(f"请求异常: {str(e)}")
       finally:
           if audio_data is not None:
               await self.shared_state.increment_image_audio_id()

   async def streaming_generate(
         self,
         session_id: str,
         keep_open: bool = False,
        ) -> Generator[Dict[str, Any], None, None]:
      """
      Omni流式生成接口
      
      Args:
          session_id: 会话ID
          mode: 模式，'simplex'(单工) 或 'duplex'(双工)
          stream: 是否流式输出，默认True
          **kwargs: 其他参数
      
      Yields:
          流式响应数据字典
      """
      try:
          # 模型如果是单工并且正在输出，则直接返回
          if self.model_type == ModelType.SIMPLEX and not self.play_end_event.is_set():
              logger.info(f"模型和前端正在输出,忽略generate")
              yield {"success": True, "type": "done"}   
              return
          # 参数验证
          if self.model_type not in [ModelType.SIMPLEX, ModelType.DUPLEX]:
              raise ValueError(f"不支持的模式: {self.model_type}，支持的模式: {ModelType.SIMPLEX}, {ModelType.DUPLEX}")
          self.model_generating_flag.set()
          # 构建请求数据
          request_data = {
              "session_id": session_id,
              "mode": self.model_type.value,
              "stream": True
          }
          if keep_open:
              # latency patch: this stream is the session's only open generate; the service keeps it open
              request_data["keep_open"] = True
          
          # 构建API URL
          api_url = f"{self.api_base_url}/omni/streaming_generate"
          
          logger.info(f"发送Omni generate请求到: {api_url}, 请求参数: {request_data}")
          
          self.play_end_event.clear()
          send_first_chunk = False
          # 流式请求
          await self.shared_state.increment_round()
          async for chunk in self.http_util.stream_post(
              url=api_url,
              json_data=request_data,
              headers={'Content-Type': 'application/json'}
          ):
            # 解析流式数据
            if not send_first_chunk:
                self.first_tts.set()
                await self.text_output_queue.put(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} - <state><generate_first_chunk>")
                send_first_chunk = True
            parsed_data = self._parse_stream_chunk(chunk)
            if parsed_data:
                yield parsed_data
      except Exception as e:
          error_msg = str(e) if str(e) else repr(e)
          logger.error(f"Omni generate请求异常: {type(e).__name__}: {error_msg}")
          raise HTTPUtilError(f"请求异常: {type(e).__name__}: {error_msg}")
      finally:
          logger.info("streaming_generate 模型输出完成")
          self.model_generating_flag.clear()

   async def streaming_break(self, session_id: str, text: str = ""):
        try:
            data = None
            if not self.model_generating_flag.is_set():
                logger.info(f"模型和前端已经输出结束,忽略break")                
            else:
                response = await self.http_util.post(
                    url=f"{self.break_url}/omni/break",
                    headers={'Content-Type': 'application/json'}
                    )
                # 回到模型聆听中
                logger.info(f"模型打断成功, 回到模型聆听中, response={response}")
                data = response['data']
                self.play_end_event.set()
            # 清空audio_output_queue
            while not self.audio_output_queue.empty():
                self.audio_output_queue.get_nowait()
            await self.text_output_queue.put("<state><session_break>")
            return data
        except Exception as e:
            logger.error(f"Omni break请求异常: {str(e)}")
        finally:
             # 问题回答结束，重置vad检测的标志位
            self.vad_stream_started = False
            self.play_end_event.set()
            await self.shared_state.increment_round()

   async def streaming_stop(self, session_id: str):
        try:
            response = await self.http_util.post(
                url=f"{self.break_url}/omni/stop",
                headers={'Content-Type': 'application/json'}
            )
            logger.info(f"Omni stop请求返回结果: {response}")
            # 重置模型为可用
            if (response['success']):
                service_manager = await get_service_manager()
                await service_manager.release_service_lock(self.inference_service.service_id, self.request.userId)
                logger.info(f"模型停止成功, 释放服务锁定: {self.inference_service.service_id}")
            return response['data']
        except Exception as e:
            logger.error(f"Omni stop请求异常: {str(e)}")
            raise HTTPUtilError(f"请求异常: {str(e)}")
            
   async def play_end(self):
        self.play_end_event.set()
        # 问题回答结束，重置vad检测的标志位
        self.vad_stream_started = False


   def _encode_audio_to_base64(self, audio_data: np.ndarray, input_sample_rate: int = 48000, output_sample_rate: int = 16000) -> str:
       """
       将音频数据编码为base64格式
       
       Args:
           audio_data: 音频numpy数组（输入采样率为 input_sample_rate）
           input_sample_rate: 输入音频采样率，默认48kHz
           output_sample_rate: 输出音频采样率，默认16kHz
       
       Returns:
           base64编码的音频数据
       """
       try:
            from scipy.signal import resample_poly
            import soundfile as sf
            import io

            # 转换为 float32 并归一化到 [-1.0, 1.0]
            if audio_data.dtype == np.int16:
                # int16 范围是 [-32768, 32767]，需要归一化
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
            
            # 重采样：从 input_sample_rate 转换到 output_sample_rate
            if input_sample_rate != output_sample_rate:
                audio_data = resample_poly(audio_data, output_sample_rate, input_sample_rate)
            
            # 裁剪到 [-1.0, 1.0] 范围（防止重采样后溢出）
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # 写入 WAV 格式
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, output_sample_rate, format='WAV', subtype='PCM_16')
            wav_bytes = wav_buffer.getvalue()
            
            # 编码为 base64
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            return audio_base64
       except Exception as e:
           logger.error(f"音频编码失败: {str(e)}")
           raise HTTPUtilError(f"音频编码失败: {str(e)}")

   def _encode_image_to_base64(self, image_data, image_format: str = "jpeg") -> str:
       """
       将图片数据编码为base64格式
       
       Args:
           image_data: 图片数据（numpy数组、字节或PIL Image对象）
           image_format: 图片格式，默认jpeg
       
       Returns:
           base64编码的图片数据URL
       """
       try:
           import io
           from PIL import Image
           
           # PIL格式映射：将常见的格式名称映射到PIL支持的格式
           format_mapping = {
               "jpg": "JPEG",
               "jpeg": "JPEG",
               "png": "PNG",
               "gif": "GIF",
               "bmp": "BMP",
               "webp": "WEBP"
           }
           
           # 获取正确的PIL格式名称
           pil_format = format_mapping.get(image_format.lower(), image_format.upper())
           
           if isinstance(image_data, np.ndarray):
               # 如果是numpy数组，需要转换为字节
               
               # 确保数组范围在[0, 255]
               if image_data.dtype != np.uint8:
                   if image_data.max() <= 1.0:
                       image_data = (image_data * 255).astype(np.uint8)
                   else:
                       image_data = image_data.astype(np.uint8)
               
               # 转换为PIL Image
               if len(image_data.shape) == 3:
                   image = Image.fromarray(image_data)
               else:
                   image = Image.fromarray(image_data, mode='L')
               
               # 转换为字节
               img_buffer = io.BytesIO()
               image.save(img_buffer, format=pil_format)
               image_bytes = img_buffer.getvalue()
               
           elif isinstance(image_data, bytes):
               image_bytes = image_data
               
           elif hasattr(image_data, 'save'):  # PIL Image对象
               # 如果是PIL Image对象，直接保存为字节
               img_buffer = io.BytesIO()
               image_data.save(img_buffer, format=pil_format)
               image_bytes = img_buffer.getvalue()
               
           else:
               raise ValueError(f"不支持的图片数据类型: {type(image_data)}")
           
           # 编码为base64
           image_base64 = base64.b64encode(image_bytes).decode('utf-8')
           
           # 返回data URL格式
           return image_base64
           
       except Exception as e:
           logger.error(f"图片编码失败: {str(e)}")
           raise HTTPUtilError(f"图片编码失败: {str(e)}")

   def _parse_stream_chunk(self, chunk: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
       """
       解析流式数据块
       
       Args:
           chunk: 数据块，可以是字符串或字典
       
       Returns:
           解析后的数据字典，其中 base64 编码的音频数据会被解码为 numpy 数组
       """
       try:
           data = None
           
           # 如果已经是字典，直接使用
           if isinstance(chunk, dict):
               data = chunk
           else:
               # 移除可能的前缀
               if chunk.startswith('data: '):
                   chunk = chunk[6:]
               
               # 检查是否是结束标记
               if chunk.strip() == '[DONE]':
                   return {'type': 'done'}
               
               # 检查是否是 done 标记的 JSON 格式
               if '"done": true' in chunk or '"done":true' in chunk:
                   return {'type': 'done'}
               
               # 尝试解析JSON
               try:
                   data = json.loads(chunk)
               except json.JSONDecodeError:
                   logger.debug(f"跳过非JSON格式的数据块")
                   return None
           
           # 解码 base64 音频数据
           if data and 'chunk_data' in data:
               chunk_data = data['chunk_data']
               if 'wav' in chunk_data and isinstance(chunk_data['wav'], str):
                   # 解码 base64 为 numpy 数组
                   wav_base64 = chunk_data['wav']
                   wav_bytes = base64.b64decode(wav_base64)
                   # 假设是 int16 格式（常见的音频格式）
                   wav_np = np.frombuffer(wav_bytes, dtype=np.int16)
                   chunk_data['wav'] = wav_np
                   logger.info(f"解码音频数据: {len(wav_np)} 样本")
           
           return data
       except Exception as e:
           logger.error(f"解析流式数据异常: {str(e)}")
           return None