"""
Proxy subclasses for fixed-point quantizers.

These exist purely to give Brevitas's handler-matching machinery a distinct
type to dispatch on. Brevitas matches handlers to proxies by *exact* type
(no_inheritance=True, see brevitas.export.manager._set_proxy_export_handler),
so without a dedicated subclass our handler would either miss our proxies or
hijack any standard Brevitas weight/activation quantizer in the same model.
"""

from brevitas.proxy.parameter_quant import (
    BiasQuantProxyFromInjector,
    WeightQuantProxyFromInjector,
)
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector


class FixedPointWeightQuantProxy(WeightQuantProxyFromInjector):
    pass


class FixedPointActQuantProxy(ActQuantProxyFromInjector):
    pass


class FixedPointBiasQuantProxy(BiasQuantProxyFromInjector):
    pass
