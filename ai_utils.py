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
        Initialize AI utility class with configuration. Separate OpenAI and local AI configs.
        """
        self.cfg = cfg
        # Separate OpenAI and LocalAI configurations
        self.openai_cfg = cfg.get("openai", {})
        self.localai_cfg = cfg.get("localai", {})

        self.local_model_path = self.localai_cfg.get(
            "local_model_path", "models/ggml-model-q4.bin"
        )
        self.local_model = None
        self.local_tokenizer = None
        self.local_capable = self._detect_environment()
        if self.local_capable:
            self._load_local_model()

    def _detect_environment(self) -> bool:
        """
        Detect if the environment can support local inference (GPU via CUDA/MPS or CPU).
        """
        min_ram = self.localai_cfg.get("min_ram", 16)  # Default: 16 GB RAM
        min_cores = self.localai_cfg.get("min_cores", 8)  # Default: 8 cores
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = psutil.cpu_count(logical=True)
        gpu_cuda = torch.cuda.is_available()
        gpu_mps = torch.backends.mps.is_available()  # Check for MPS on macOS
        gpu_type = "CUDA" if gpu_cuda else "MPS" if gpu_mps else "None"

        logger.info(
            f"Detected system specs: {ram_gb:.1f} GB RAM, {cpu_count} CPU cores, GPU: {gpu_type}"
        )

        # Decide local AI capability based on thresholds and detected specs
        if (ram_gb >= min_ram and cpu_count >= min_cores) or gpu_cuda or gpu_mps:
            logger.info("Local environment is capable of AI inference.")
            return True
        logger.warning("Local environment does not meet AI inference requirements for local model.")
        return False

    def _load_local_model(self):
        """
        Load the local AI model (GPU with CUDA/MPS or CPU quantized model).
        """
        try:
            model_name = self.localai_cfg.get("local_model", "EleutherAI/gpt-neo-2.7B")

            if torch.cuda.is_available():
                logger.info("Loading GPU model with CUDA...")
                self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", torch_dtype=torch.float16
                )
            elif torch.backends.mps.is_available():
                logger.info("Loading GPU model with MPS...")
                self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", torch_dtype=torch.float32  # MPS requires float32
                )
            else:
                logger.info("No GPU detected. Attempting to use Local CPU model.")
                if not os.path.exists(self.local_model_path):
                    logger.info("Local model not found. Attempting to download...")
                    self._download_model()

                logger.info(f"Using binary model at: {self.local_model_path}")
                if not os.path.exists(self.local_model_path):
                    raise FileNotFoundError(
                        f"Local model binary not found: {self.local_model_path}"
                    )
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self.local_capable = False  # Fallback to OpenAI API if local fails

    def _download_model(self):
        """
        Downloads a pre-trained model if missing, saving to the configured local_model_path.
        """
        model_url = self.localai_cfg.get("model_url")
        if not model_url:
            raise ValueError("No model URL configured for local AI.")

        save_path = Path(self.local_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading model from {model_url}...")
            import requests

            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            with save_path.open("wb") as model_file:
                for chunk in response.iter_content(chunk_size=8192):
                    model_file.write(chunk)

            logger.info(f"Model successfully downloaded to {self.local_model_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to download model from {model_url}: {e}")
            raise

    def _run_local_inference(self, prompt: str) -> str:
        """
        Perform local inference on the given prompt using GPU or CPU-based models.
        """
        if not self.local_model and not self.local_tokenizer:
            raise RuntimeError("Local model is not properly initialized.")

        logger.info("Running inference with local model...")
        try:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            inputs = self.local_tokenizer(prompt, return_tensors="pt").to(device)
            outputs = self.local_model.generate(**inputs, max_length=200)
            return self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Local inference failed: {e}")
            raise RuntimeError(f"Local inference failed: {e}")

    def _call_openai_api(self, prompt: str) -> str:
        """
        Perform API-based inference via OpenAI configuration.
        """
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY", self.openai_cfg.get("api_key"))
            model_name = self.openai_cfg.get("model", "gpt-3.5-turbo")
            max_tokens = self.openai_cfg.get("max_tokens", 16000)
            temperature = self.openai_cfg.get("temperature", 0.7)

            logger.info("Performing fallback to OpenAI API...")
            response = openai.ChatCompletion.create(
                model=model_name,
                temperature=temperature,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return "⚠️ OpenAI API fallback failed."

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using local AI if available, otherwise fall back to OpenAI.
        """
        try:
            if self.local_capable:
                return self._run_local_inference(prompt)
            else:
                return self._call_openai_api(prompt)
        except Exception as e:
            logger.warning(f"⚠️ Falling back to OpenAI API after local error: {e}")
            return self._call_openai_api(prompt)
