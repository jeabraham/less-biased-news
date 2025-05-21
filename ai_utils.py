import logging
import os
import subprocess
from pathlib import Path
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

logger = logging.getLogger(__name__)


class AIUtils:
    def __init__(self, cfg):
        """
        Initialize AI utility class with configuration. Check local vs API readiness.
        """
        self.cfg = cfg
        self.local_capable = self._detect_environment()
        self.local_model = None
        self.local_tokenizer = None
        if self.local_capable:
            self._load_local_model()

    def _detect_environment(self) -> bool:
        """
        Detect if the environment can support local inference (GPU via CUDA/ROCm or powerful CPU).
        """
        min_ram = self.cfg.get("min_ram", 16)  # Minimum RAM in GB
        min_cores = self.cfg.get("min_cores", 8)  # Minimum logical cores
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = psutil.cpu_count(logical=True)
        gpu_cuda = torch.cuda.is_available()
        gpu_rocm = torch.backends.mps.is_available() or torch.backends.hip.is_available()  # ROCm check

        logger.info(
            f"Detected system specs: {ram_gb:.1f} GB RAM, {cpu_count} CPU cores, "
            f"CUDA GPU: {gpu_cuda}, ROCm GPU: {gpu_rocm}"
        )

        # Decide local AI capability based on thresholds and detected specs
        if (ram_gb >= min_ram and cpu_count >= min_cores) or (gpu_cuda or gpu_rocm):
            logger.info("Local environment is capable of AI inference.")
            return True
        logger.warning("Local environment does not meet AI inference requirements for local model.")
        return False

    def _load_local_model(self):
        """
        Load the local AI model (GPU with CUDA/ROCm or CPU quantized model).
        """
        try:
            if torch.cuda.is_available():
                logger.info("Loading GPU model with CUDA...")
                model_name = self.cfg.get("local_model", "EleutherAI/gpt-neo-2.7B")  # Update as required
                self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", torch_dtype=torch.float16
                )
            elif torch.backends.mps.is_available() or torch.backends.hip.is_available():
                logger.info("Loading GPU model with ROCm support...")
                model_name = self.cfg.get("local_model", "EleutherAI/gpt-neo-2.7B")
                self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", torch_dtype=torch.float16
                )
            else:
                logger.info("Loading quantized CPU model using llama.cpp...")
                self.local_model_path = self.cfg.get("local_model_path", "path/to/llama/cpp/model.bin")
                if not os.path.exists(self.local_model_path):
                    raise ValueError(f"Model file not found: {self.local_model_path}")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self.local_capable = False  # Fallback to OpenAI API

    def _run_local_inference(self, prompt: str) -> str:
        """
        Perform local inference on the given prompt using GPU or CPU-based models.
        """
        if torch.cuda.is_available() or torch.backends.mps.is_available() or torch.backends.hip.is_available():
            logger.info("Running inference on GPU with local model...")
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "hip"
            inputs = self.local_tokenizer(prompt, return_tensors="pt").to(device)
            outputs = self.local_model.generate(**inputs, max_length=200)
            return self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            logger.info("Running inference with CPU quantized model (llama.cpp)...")
            result = subprocess.run(
                ["llama-cpp", self.local_model_path, "--threads", str(psutil.cpu_count(logical=True))],
                input=prompt,
                text=True,
                capture_output=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"llama.cpp inference failed: {result.stderr}")
            return result.stdout.strip()

    def _call_openai_api(self, prompt: str) -> str:
        """
        Perform API-based inference via OpenAI in case of local failure.
        """
        try:
            logger.info("Performing fallback to OpenAI API...")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            model_name = self.cfg["openai"]["model"]
            response = openai.ChatCompletion.create(
                model=model_name,
                temperature=self.cfg["openai"].get("temperature", 0.7),
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return "⚠️ OpenAI API fallback failed."

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response, using local inference or falling back to OpenAI's API.
        """
        try:
            if self.local_capable:
                return self._run_local_inference(prompt)
            else:
                return self._call_openai_api(prompt)
        except Exception as e:
            logger.warning(f"⚠️ Falling back to OpenAI API after error: {e}")
            return self._call_openai_api(prompt)
