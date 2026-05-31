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
        
        # Global flag to force all quantizers to re-run their search/calibration.
        # NOTE: `force_recalibration` is consumed by the first quantizer in the
        # forward order — kept only for back-compat. New code should use the
        # generation counter (`calibration_generation`) instead, which lets every
        # registered quantizer see "this generation hasn't been calibrated yet"
        # and recalibrate once even if many run within a single forward pass.
        self.force_recalibration = False
        self.calibration_generation = 0
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
        self.calibration_generation = 0
        self.quantization_start_gap = 0
        self.quantizers.clear()
        self._id_counter = 0
        self._inference_sequence_id_counter = 0

    @property
    def is_quantizing_everything_fully(self):
        for quant in self.quantizers.values():
            if quant.annealing_alpha != 1.0:
                return False
        return True

    @property
    def is_not_quantizing_at_all(self):
        for quant in self.quantizers.values():
            if quant.annealing_alpha != 0.0:
                return False
        return True

    def stop_quantization_for_n_inferences(self, n):
        for quant in self.quantizers.values():
            quant.inference_counter = -n

    def set_annealing_for_n_inferences(self, n):
        alpha_step = 1.0/n
        for quant in self.quantizers.values():
            quant.annealing_alpha.data.fill_(0)
            quant.annealing_alpha_step = alpha_step

    def disable_quantization(self):
        """Disable quantization by setting annealing_alpha and annealing_alpha_step to zero for all registered quantizers."""
        for quant in self.quantizers.values():
            quant.annealing_alpha.data.fill_(0.0)
            quant.annealing_alpha_step = 0.0

    def enable_quantization(self):
        """Enable quantization by setting annealing_alpha to one and annealing_alpha_step to 0.1 for all registered quantizers."""
        for quant in self.quantizers.values():
            quant.annealing_alpha.data.fill_(1.0)
            quant.annealing_alpha_step = 0.1

    def force_alpha_one(self):
        """
        Pin every quantizer's annealing_alpha to 1.0 (full quantization, no soft mix)
        and zero the step so it can't drift. Used by the bit-width annealing path,
        which doesn't want any mixing — only the effective bit-width should anneal.
        """
        for quant in self.quantizers.values():
            quant.annealing_alpha.data.fill_(1.0)
            quant.annealing_alpha_step = 0.0

    def set_bit_width(self, effective_bw: int):
        """
        Update every registered quantizer's `effective_bit_width` buffer and
        bump the global calibration generation so every quantizer (not just
        the first to run) re-calibrates against the new grid on its next
        forward.
        """
        for quant in self.quantizers.values():
            if hasattr(quant, 'effective_bit_width'):
                quant.effective_bit_width.fill_(int(effective_bw))
        self.calibration_generation += 1

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
