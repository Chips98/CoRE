

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
import aiohttp
import openai


class LLMClient:
    """异步LLM客户端，支持多种API类型"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM客户端

        参数:
            config: LLM配置字典
        """
        self.config = config
        self.logger = logging.getLogger("Method")

        # 基本参数
        self.model_name = config["model_name"]
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2000)
        self.timeout = config.get("timeout", 30)

        # 检测API类型
        self.api_type = self._detect_api_type()

        # 根据API类型创建客户端
        self.client = None
        self.session = None

        if self.api_type == "vllm":
            # VLLM使用aiohttp直接调用
            self.logger.info("使用VLLM API客户端")
        elif self.api_type == "openrouter":
            # OpenRouter使用OpenAI客户端，需要特殊headers
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self.logger.info("使用OpenRouter API客户端")
        else:
            # OpenAI和Dashscope使用OpenAI客户端
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.logger.info(f"使用OpenAI兼容API客户端 - 类型: {self.api_type}")

        self.logger.info(f"LLM客户端初始化完成 - 模型: {self.model_name}, 类型: {self.api_type}")

    def _detect_api_type(self) -> str:
        """检测API类型"""
        if self.base_url:
            # 本地部署的API（VLLM）- 检查常用的本地端口
            if any(port in self.base_url for port in ['7862']):
                return "vllm"
            # OpenRouter API
            elif "openrouter.ai" in self.base_url:
                return "openrouter"
            # OpenAI官方API
            elif "openai.com" in self.base_url:
                return "openai"
            # 其他兼容API（Dashscope等）
            else:
                return "dashscope"
        else:
            # 默认为dashscope
            return "dashscope"

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        生成响应

        参数:
            prompt: 输入提示
            **kwargs: 额外的生成参数

        返回:
            生成的响应文本
        """
        if self.api_type == "vllm":
            return await self._generate_vllm_response(prompt, **kwargs)
        else:
            return await self._generate_openai_response(prompt, **kwargs)

    async def one_chat(self, messages: List[Dict[str, str]], temperature: float = None, json_mode: bool = False, **kwargs) -> str:

        if self.api_type == "vllm":
            return await self._chat_vllm(messages, temperature, json_mode, **kwargs)
        else:
            return await self._chat_openai(messages, temperature, json_mode, **kwargs)

    async def _generate_openai_response(self, prompt: str, **kwargs) -> str:
        """生成OpenAI兼容API响应"""
        # 合并参数
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "timeout": self.timeout
        }

        # 添加seed参数（如果提供）
        if "seed" in kwargs and kwargs["seed"] is not None:
            params["seed"] = kwargs["seed"]

        try:
            # 调用API
            response = await self.client.chat.completions.create(**params)

            # 提取响应内容
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content
                if result is None:
                    result = ""
                else:
                    result = result.strip()
            else:
                result = ""
                self.logger.warning("LLM返回空响应")

            # 应用内容过滤
            if result:
                result = self._extract_real_answer(result)

            # 记录响应预览
            preview = result[:200] + "..." if len(result) > 200 else result
            self.logger.debug(f"LLM响应预览: {preview}")

            return result

        except openai.AuthenticationError as e:
            self.logger.error(f"LLM认证失败: {e}")
            raise
        except openai.RateLimitError as e:
            self.logger.error(f"LLM速率限制: {e}")
            raise
        except openai.APITimeoutError as e:
            self.logger.error(f"LLM请求超时: {e}")
            raise
        except openai.APIConnectionError as e:
            self.logger.error(f"LLM连接错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"LLM生成响应时发生错误: {e}")
            raise

    async def _generate_vllm_response(self, prompt: str, **kwargs) -> str:
        """生成VLLM API响应"""
        # 准备请求数据
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }

        # 添加seed参数（如果提供）
        if "seed" in kwargs and kwargs["seed"] is not None:
            data["seed"] = kwargs["seed"]

        # 准备请求头
        headers = {"Content-Type": "application/json"}

        # 构建API端点URL
        base_url = self.base_url.rstrip('/')
        # 如果 base_url 已经包含 /v1，则只需添加 /chat/completions
        if base_url.endswith('/v1'):
            url = f"{base_url}/chat/completions"
        elif not base_url.endswith('/chat/completions'):
            url = f"{base_url}/v1/chat/completions"
        else:
            url = base_url

        # 创建会话（如果还没有）
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(
                url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"VLLM API请求失败，状态码: {response.status}, 错误: {error_text}")

                response_data = await response.json()

                # 提取响应内容
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    choice = response_data['choices'][0]
                    if 'message' in choice:
                        result = choice['message'].get('content', '')
                        if result is None:
                            result = ""
                        else:
                            result = result.strip()
                    else:
                        result = ""
                else:
                    result = ""

                # 应用内容过滤
                if result:
                    result = self._extract_real_answer(result)

                # 记录响应预览
                preview = result[:200] + "..." if len(result) > 200 else result
                self.logger.debug(f"VLLM响应预览: {preview}")

                return result

        except Exception as e:
            self.logger.error(f"VLLM API调用失败: {e}")
            raise

    async def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "timeout": self.timeout,
        }

        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if "seed" in kwargs and kwargs["seed"] is not None:
            params["seed"] = kwargs["seed"]

        try:
            response = await self.client.chat.completions.create(**params)
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content
                result = "" if result is None else result.strip()
            else:
                result = ""
                self.logger.warning("LLM返回空响应")

            if result:
                result = self._extract_real_answer(result)
            return result
        except Exception as e:
            self.logger.error(f"OpenAI兼容聊天调用失败: {e}")
            raise

    async def _chat_vllm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if json_mode:
            data["response_format"] = {"type": "json_object"}
        if "seed" in kwargs and kwargs["seed"] is not None:
            data["seed"] = kwargs["seed"]

        headers = {"Content-Type": "application/json"}
        base_url = self.base_url.rstrip('/')
        if base_url.endswith('/v1'):
            url = f"{base_url}/chat/completions"
        elif not base_url.endswith('/chat/completions'):
            url = f"{base_url}/v1/chat/completions"
        else:
            url = base_url

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.post(
                url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"VLLM API请求失败，状态码: {response.status}, 错误: {error_text}")

                response_data = await response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    choice = response_data['choices'][0]
                    if 'message' in choice:
                        result = choice['message'].get('content', '')
                        result = "" if result is None else result.strip()
                    else:
                        result = ""
                else:
                    result = ""

                if result:
                    result = self._extract_real_answer(result)
                return result
        except Exception as e:
            self.logger.error(f"VLLM聊天调用失败: {e}")
            raise

    def _extract_real_answer(self, response_text: str) -> str:
        """
        从模型输出中提取真实回答内容

        参数:
            response_text: 原始模型输出

        返回:
            提取后的真实回答内容
        """
        if not response_text:
            return response_text

        # 首先处理Qwen3模型的"think"分隔符
        if "</think>" in response_text:
            parts = response_text.split("</think>")
            if len(parts) > 1:
                # 取最后一个"</think>"之后的内容
                real_content = parts[-1].strip()
                # 移除开头的换行符
                real_content = real_content.lstrip('\n')
                response_text = real_content
                # print(f"response_text:{response_text}")
    
        return response_text

    async def test_connection(self) -> bool:
        """
        测试LLM连接

        返回:
            连接是否成功
        """
        try:
            test_prompt = "请回答：1+1等于几？"
            response = await self.generate(test_prompt, max_tokens=10)

            if response and len(response) > 0:
                self.logger.info("LLM连接测试成功")
                return True
            else:
                self.logger.error("LLM连接测试失败：收到空响应")
                return False

        except Exception as e:
            self.logger.error(f"LLM连接测试失败: {e}")
            return False

    async def close(self):
        """关闭客户端连接"""
        try:
            if self.client:
                await self.client.close()
            if self.session:
                await self.session.close()
            self.logger.info("LLM客户端已关闭")
        except Exception as e:
            self.logger.error(f"关闭LLM客户端时发生错误: {e}")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
