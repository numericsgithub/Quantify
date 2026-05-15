class QuantizerManager:
    """
    Singleton manager object for coordinating quantizer instances across the entire project.
    Ensures a single shared reference exists for global coordination, such as forcing 
    re-calibration or tracking global quantization statistics.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization on subsequent calls to __new__
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        # Global flag to force all quantizers to re-run their search/calibration
        self.force_recalibration = False
        self.quantization_is_enabled_globally = True
        self.quantization_start_gap = 0
        # Registry to keep track of all active quantizer instances {id: quantizer}
        self.quantizers = {}
        # Counter to generate unique identifiers
        self._id_counter = 0
        self._inference_sequence_id_counter = 0

    def reset(self):
        """
        Reset the manager's internal state. Useful for testing or restarting experiments.
        """
        self.force_recalibration = False
        self.quantization_is_enabled_globally = True
        self.quantization_start_gap = 0
        self.quantizers.clear()
        self._id_counter = 0
        self._inference_sequence_id_counter = 0

    def stop_quantization_for_n_inferences(self, n):
        for quant in self.quantizers.values():
            quant.inference_counter = -n

    def set_annealing_for_n_inferences(self, n):
        alpha_step = 1.0/n
        for quant in self.quantizers.values():
            quant.annealing_alpha = 0
            quant.annealing_alpha_step = alpha_step

    def register_quantizer(self, quantizer):
        """
        Registers a quantizer instance with the manager and assigns it a unique ID.
        """
        if quantizer in self.quantizers.values():
            return

        qid = f"quant_{self._id_counter}"
        self.quantizers[qid] = quantizer
        
        # Assign the unique ID back to the quantizer object for easy reference
        quantizer.quant_id = qid
        quantizer.inference_sequence_id = -1

        self._id_counter += 1

    def get_inference_sequence_id(self):
        self._inference_sequence_id_counter += 1
        return self._inference_sequence_id_counter - 1

    def trigger_global_recalibration(self):
        """Sets the flag to force all quantizers to re-calibrate on next forward."""
        self.force_recalibration = True

    def reset_global_flag(self):
        """Resets the global recalibration flag."""
        self.force_recalibration = False
