#include <Python.h>

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(feather_ops, m)
{
}

extern "C" PyObject* PyInit__C(void)
{
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT,
        "_C",
        nullptr,
        -1,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };
    return PyModule_Create(&module);
}
