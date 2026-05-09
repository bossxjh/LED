class _LazyAdapter:
    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name

    def __call__(self, *args, **kwargs):
        module = __import__(self.module_name, fromlist=[self.class_name])
        cls = getattr(module, self.class_name)
        return cls(*args, **kwargs)


MODEL_ADAPTERS = {
    "clip": _LazyAdapter("dataeval.models.clip_adapter", "CLIPAdapter"),
    "openvla": _LazyAdapter("dataeval.models.openvla_adapter", "OpenVLAAdapter"),
    "pi0.5": _LazyAdapter("dataeval.models.pi0_5_adapter", "Pi05Adapter"),
    "diffusion policy": _LazyAdapter("dataeval.models.diffusion_policy_adapter", "DPResNetAdapter"),
}
