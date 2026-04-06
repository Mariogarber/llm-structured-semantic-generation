# Verificar información de la GPU
import torch
import time
import logging
import threading
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict, deque
import json
import os

def check_gpu_info():
    if torch.cuda.is_available():
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            total_memory = gpu_props.total_memory / (1024**3)  # GB
            
            print(f"\n--- GPU {i}: {gpu_props.name} ---")
            print(f"Memoria total: {total_memory:.2f} GB")
            print(f"Memoria disponible: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB reservada")
            print(f"Memoria libre: {(gpu_props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3):.2f} GB")
            
            # Información adicional
            print(f"Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"Multiprocessors: {gpu_props.multi_processor_count}")
    else:
        print("CUDA no está disponible")

# Calculadora de memoria para modelos
def estimate_model_memory(num_parameters_billions, precision="float16"):
    """
    Estima la memoria necesaria para cargar un modelo
    
    Args:
        num_parameters_billions: Número de parámetros en billones (B)
        precision: Tipo de precisión ("float32", "float16", "int8", "int4")
    """
    bytes_per_param = {
        "float32": 4,  # 32 bits = 4 bytes
        "float16": 2,  # 16 bits = 2 bytes
        "bfloat16": 2, # 16 bits = 2 bytes
        "int8": 1,     # 8 bits = 1 byte
        "int4": 0.5    # 4 bits = 0.5 bytes
    }
    
    if precision not in bytes_per_param:
        precision = "float16"
    
    # Memoria base del modelo
    model_memory_gb = (num_parameters_billions * 1e9 * bytes_per_param[precision]) / (1024**3)
    
    # Agregar overhead (gradientes, optimizador, activaciones)
    # Para inferencia: ~20-30% adicional
    # Para entrenamiento: ~3-4x más memoria
    inference_memory_gb = model_memory_gb * 1.3
    training_memory_gb = model_memory_gb * 4
    
    return {
        "model_size_gb": model_memory_gb,
        "inference_memory_gb": inference_memory_gb,
        "training_memory_gb": training_memory_gb,
        "precision": precision
    }

# Ejemplos de modelos populares
models = {
    "GPT-2 Small": 0.124,      # 124M parámetros
    "GPT-2 Medium": 0.355,     # 355M parámetros  
    "GPT-2 Large": 0.774,      # 774M parámetros
    "GPT-2 XL": 1.5,           # 1.5B parámetros
    "Mistral-7B": 7.0,         # 7B parámetros
    "Llama-7B": 7.0,           # 7B parámetros
    "Llama-13B": 13.0,         # 13B parámetros
    "Llama-30B": 30.0,         # 30B parámetros
    "Llama-65B": 65.0,         # 65B parámetros
}


# =======================================================================================
# MONITORIZACIÓN AVANZADA DE MEMORIA GPU PARA FINE-TUNING
# =======================================================================================

class GPUMemoryMonitor:
    """Monitor avanzado de memoria GPU con logging y alertas en tiempo real"""
    
    def __init__(self, device_id=0, log_interval=5, memory_threshold=0.85):
        self.device_id = device_id
        self.log_interval = log_interval  # segundos
        self.memory_threshold = memory_threshold  # porcentaje de alerta
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history = deque(maxlen=1000)  # Últimas 1000 mediciones
        self.alerts = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f'GPUMemoryMonitor_GPU{device_id}')
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible")
            
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"GPU {device_id} no encontrada")
    
    def get_memory_stats(self):
        """Obtiene estadísticas detalladas de memoria GPU"""
        if not torch.cuda.is_available():
            return None
            
        props = torch.cuda.get_device_properties(self.device_id)
        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        total = props.total_memory
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'device_id': self.device_id,
            'device_name': props.name,
            'total_memory_gb': total / (1024**3),
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'free_gb': (total - reserved) / (1024**3),
            'allocated_percent': (allocated / total) * 100,
            'reserved_percent': (reserved / total) * 100,
            'memory_efficiency': (allocated / reserved * 100) if reserved > 0 else 0
        }
        
        return stats
    
    def start_monitoring(self):
        """Inicia el monitoreo en tiempo real"""
        if self.monitoring:
            self.logger.warning("El monitoreo ya está activo")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Monitoreo iniciado para GPU {self.device_id}")
    
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.logger.info("Monitoreo detenido")
    
    def _monitor_loop(self):
        """Loop principal de monitoreo"""
        while self.monitoring:
            try:
                stats = self.get_memory_stats()
                if stats:
                    self.memory_history.append(stats)
                    
                    # Verificar alertas
                    if stats['reserved_percent'] > self.memory_threshold * 100:
                        alert = f"  ALERTA: Uso de memoria GPU {stats['reserved_percent']:.1f}% (límite: {self.memory_threshold*100}%)"
                        self.alerts.append({'timestamp': stats['timestamp'], 'message': alert})
                        self.logger.warning(alert)
                    
                    # Log periódico
                    self.logger.info(f"GPU {self.device_id}: {stats['allocated_percent']:.1f}% asignada, {stats['reserved_percent']:.1f}% reservada")
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                self.logger.error(f"Error en monitoreo: {e}")
                time.sleep(1)
    
    def get_memory_summary(self):
        """Obtiene resumen de memoria con estadísticas"""
        if not self.memory_history:
            return None
            
        recent_stats = list(self.memory_history)[-10:]  # Últimas 10 mediciones
        
        allocated_values = [s['allocated_percent'] for s in recent_stats]
        reserved_values = [s['reserved_percent'] for s in recent_stats]
        
        summary = {
            'current': recent_stats[-1] if recent_stats else None,
            'avg_allocated_10min': sum(allocated_values) / len(allocated_values),
            'max_allocated': max(allocated_values),
            'avg_reserved_10min': sum(reserved_values) / len(reserved_values),
            'max_reserved': max(reserved_values),
            'measurements_count': len(self.memory_history),
            'alerts_count': len(self.alerts),
            'latest_alerts': self.alerts[-5:] if self.alerts else []
        }
        
        return summary
    
    def save_memory_log(self, filepath):
        """Guarda el historial de memoria en un archivo JSON"""
        log_data = {
            'device_info': {
                'device_id': self.device_id,
                'device_name': torch.cuda.get_device_properties(self.device_id).name,
                'monitoring_config': {
                    'log_interval': self.log_interval,
                    'memory_threshold': self.memory_threshold
                }
            },
            'memory_history': list(self.memory_history),
            'alerts': self.alerts,
            'summary': self.get_memory_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Log de memoria guardado en: {filepath}")


@contextmanager
def monitor_gpu_memory(device_id=0, log_interval=10, memory_threshold=0.85, save_log=None):
    """
    Context manager para monitoreo automático de memoria durante operaciones
    
    Args:
        device_id: ID de la GPU a monitorizar
        log_interval: Intervalo de logging en segundos
        memory_threshold: Umbral de alerta (0-1)
        save_log: Ruta para guardar logs (opcional)
    
    Example:
        with monitor_gpu_memory(device_id=0, save_log="training_memory.json"):
            # Tu código de entrenamiento aquí
            model.train()
    """
    monitor = GPUMemoryMonitor(device_id, log_interval, memory_threshold)
    
    try:
        monitor.start_monitoring()
        yield monitor
    finally:
        monitor.stop_monitoring()
        if save_log:
            monitor.save_memory_log(save_log)


def get_memory_usage():
    """Obtiene uso actual de memoria en todas las GPUs disponibles"""
    if not torch.cuda.is_available():
        return {"error": "CUDA no disponible"}
    
    usage = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = props.total_memory / (1024**3)
        
        usage[f"GPU_{i}"] = {
            "name": props.name,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2),
            "usage_percent": round((reserved / total) * 100, 1)
        }
    
    return usage


def clear_gpu_memory(device_id=None, verbose=True):
    """
    Limpia la memoria GPU de forma agresiva
    
    Args:
        device_id: GPU específica o None para todas
        verbose: Mostrar información de limpieza
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA no disponible")
        return False
    
    devices = [device_id] if device_id is not None else range(torch.cuda.device_count())
    
    for dev_id in devices:
        if verbose:
            before = torch.cuda.memory_reserved(dev_id) / (1024**3)
        
        # Limpiar caché
        torch.cuda.empty_cache()
        
        # Forzar garbage collection
        import gc
        gc.collect()
        
        # Sincronizar para asegurar limpieza completa
        torch.cuda.synchronize(dev_id)
        
        if verbose:
            after = torch.cuda.memory_reserved(dev_id) / (1024**3)
            cleared = before - after
            print(f"GPU {dev_id}: Liberados {cleared:.2f} GB (de {before:.2f} GB a {after:.2f} GB)")
    
    return True


def memory_cleanup_callback():
    """Callback para limpieza automática de memoria durante entrenamiento"""
    def cleanup():
        clear_gpu_memory(verbose=False)
        import gc
        gc.collect()
    return cleanup


@contextmanager
def auto_memory_cleanup(cleanup_interval_steps=100):
    """
    Context manager que ejecuta limpieza automática cada N pasos
    
    Args:
        cleanup_interval_steps: Número de pasos entre limpiezas
    
    Example:
        with auto_memory_cleanup(cleanup_interval_steps=50):
            for step, batch in enumerate(dataloader):
                # código de entrenamiento
                pass
    """
    step_counter = [0]  # Lista para modificar en nested function
    
    def track_step():
        step_counter[0] += 1
        if step_counter[0] % cleanup_interval_steps == 0:
            clear_gpu_memory(verbose=False)
    
    try:
        yield track_step
    finally:
        clear_gpu_memory(verbose=True)


def estimate_batch_memory(model, batch_size, sequence_length, vocab_size=50000, dtype=torch.float16):
    """
    Estima memoria necesaria para un batch específico
    
    Args:
        model: Modelo PyTorch o número de parámetros en billones
        batch_size: Tamaño del batch
        sequence_length: Longitud de secuencia
        vocab_size: Tamaño del vocabulario
        dtype: Tipo de datos (torch.float16, torch.float32, etc.)
    """
    bytes_per_element = 2 if dtype == torch.float16 else 4
    
    # Si es un número, asumir que son parámetros en billones
    if isinstance(model, (int, float)):
        model_params = model * 1e9
    else:
        # Contar parámetros del modelo
        model_params = sum(p.numel() for p in model.parameters())
    
    # Estimaciones (aproximadas)
    model_memory = model_params * bytes_per_element
    
    # Activaciones para transformer (aproximado)
    # Hidden states + attention weights + feed forward
    hidden_dim = int((model_params / (12 * sequence_length * vocab_size))**0.5)  # Estimación rough
    activations_per_sample = sequence_length * hidden_dim * 12 * 4  # 4 copias típicas
    activations_memory = batch_size * activations_per_sample * bytes_per_element
    
    # Gradientes
    gradients_memory = model_memory
    
    # Optimizador (Adam: momentos + varianzas)
    optimizer_memory = model_memory * 2
    
    total_memory = model_memory + activations_memory + gradients_memory + optimizer_memory
    
    return {
        "model_memory_gb": model_memory / (1024**3),
        "activations_memory_gb": activations_memory / (1024**3),
        "gradients_memory_gb": gradients_memory / (1024**3),
        "optimizer_memory_gb": optimizer_memory / (1024**3),
        "total_memory_gb": total_memory / (1024**3),
        "batch_size": batch_size,
        "sequence_length": sequence_length
    }


def optimize_batch_size_for_gpu(model, sequence_length=512, target_memory_usage=0.8, device_id=0):
    """
    Encuentra el batch size óptimo para la GPU disponible
    
    Args:
        model: Modelo o número de parámetros
        sequence_length: Longitud de secuencia
        target_memory_usage: Porcentaje objetivo de uso de memoria (0-1)
        device_id: ID de la GPU
    
    Returns:
        dict: Información del batch size óptimo
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA no disponible"}
    
    props = torch.cuda.get_device_properties(device_id)
    available_memory = props.total_memory * target_memory_usage
    
    # Búsqueda binaria del batch size óptimo
    min_batch = 1
    max_batch = 512
    optimal_batch = 1
    
    while min_batch <= max_batch:
        test_batch = (min_batch + max_batch) // 2
        memory_estimate = estimate_batch_memory(model, test_batch, sequence_length)
        
        if memory_estimate["total_memory_gb"] * (1024**3) <= available_memory:
            optimal_batch = test_batch
            min_batch = test_batch + 1
        else:
            max_batch = test_batch - 1
    
    final_estimate = estimate_batch_memory(model, optimal_batch, sequence_length)
    
    return {
        "optimal_batch_size": optimal_batch,
        "memory_usage_gb": final_estimate["total_memory_gb"],
        "memory_usage_percent": (final_estimate["total_memory_gb"] * (1024**3) / props.total_memory) * 100,
        "gpu_total_memory_gb": props.total_memory / (1024**3),
        "sequence_length": sequence_length,
        "breakdown": final_estimate
    }


class TrainingMemoryCallback:
    """Callback para monitoreo de memoria durante entrenamiento con HuggingFace Transformers"""
    
    def __init__(self, log_frequency=10, cleanup_frequency=100, memory_threshold=0.9):
        self.log_frequency = log_frequency
        self.cleanup_frequency = cleanup_frequency
        self.memory_threshold = memory_threshold
        self.step = 0
        self.memory_logs = []
        
    def on_step_begin(self, args=None, state=None, control=None, **kwargs):
        """Callback ejecutado al inicio de cada step"""
        self.step += 1
        
        # Log periódico
        if self.step % self.log_frequency == 0:
            usage = get_memory_usage()
            self.memory_logs.append({
                'step': self.step,
                'timestamp': datetime.now().isoformat(),
                'memory_usage': usage
            })
            
            # Verificar alertas
            for gpu_id, gpu_info in usage.items():
                if isinstance(gpu_info, dict) and gpu_info.get('usage_percent', 0) > self.memory_threshold * 100:
                    print(f"  Step {self.step}: {gpu_id} usando {gpu_info['usage_percent']:.1f}% de memoria")
        
        # Limpieza periódica
        if self.step % self.cleanup_frequency == 0:
            clear_gpu_memory(verbose=False)
    
    def get_memory_report(self):
        """Genera reporte de memoria del entrenamiento"""
        if not self.memory_logs:
            return "No hay datos de memoria registrados"
        
        report = f"""
=== REPORTE DE MEMORIA DEL ENTRENAMIENTO ===
Pasos totales monitoreados: {len(self.memory_logs)}
Frequency de logging: cada {self.log_frequency} pasos
Frequency de limpieza: cada {self.cleanup_frequency} pasos

Uso de memoria por GPU:
"""
        
        # Analizar tendencias de memoria
        for gpu_id in self.memory_logs[0]['memory_usage'].keys():
            if gpu_id.startswith('GPU_'):
                usage_values = [log['memory_usage'][gpu_id]['usage_percent'] 
                               for log in self.memory_logs 
                               if gpu_id in log['memory_usage']]
                
                if usage_values:
                    avg_usage = sum(usage_values) / len(usage_values)
                    max_usage = max(usage_values)
                    
                    report += f"{gpu_id}: Promedio {avg_usage:.1f}%, Máximo {max_usage:.1f}%\n"
        
        return report
    
    def save_memory_logs(self, filepath):
        """Guarda logs de memoria en archivo JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.memory_logs, f, indent=2)
        print(f"Logs de memoria guardados en: {filepath}")
