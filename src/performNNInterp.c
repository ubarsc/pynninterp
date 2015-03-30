/*
 * performNNInterp.cpp
 *
 *  Created by Pete Bunting on 27/03/2015.
 *
 * This file is part of PyNNInterp
 * Copyright (C) 2015 John Armston, Pete Bunting, Neil Flood, Sam Gillingham
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <errno.h>

/*
#include "nan.h"
#include "minell.h"
#include "nn.h"
*/

/* An exception object for this module */
/* created in the init function */
struct PyNNInterpState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct PyNNInterpState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct PyNNInterpState _state;
#endif


int notDoubleVector1D(PyArrayObject *vec)
{
    if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)
    {
        PyErr_SetString(PyExc_ValueError, "In notDoubleVector1D: array must be of type Float and 1 dimensional (n).");
        return 1;
    }
    return 0;
}

int notDoubleVector2D(PyArrayObject *vec)
{
    if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 2)
    {
        PyErr_SetString(PyExc_ValueError, "In notDoubleVector2D: array must be of type Float and 2 dimensional (n).");
        return 1;
    }
    return 0;
}

static PyObject *PyNNInterp_interpNNGrid(PyObject *self, PyObject *args, PyObject *keywds)
{
    printf ("Entered of function.\n");
    PyArrayObject *inX;
    PyArrayObject *inY;
    PyArrayObject *inValues;
    PyArrayObject *xGrid;
    PyArrayObject *yGrid;
    printf ("Created Variables.\n");
    
    static char *kwlist[] = {"x", "y", "values", "xGrid", "yGrid", NULL};
    
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOO:interpNNGrid", kwlist, &inX, &inY, &inValues, &xGrid, &yGrid))
    {
        return NULL;
    }
    
    if(inX == NULL || notDoubleVector1D(inX))
    {
        PyErr_SetString(GETSTATE(self)->error, "inX must be type double (numpy.float64) and 1D");
        return NULL;
    }
    size_t nInX = inX->dimensions[0];
    
    if(inY == NULL || notDoubleVector1D(inY))
    {
        PyErr_SetString(GETSTATE(self)->error, "inY must be type double (numpy.float64) and 1D");
        return NULL;
    }
    size_t nInY = inY->dimensions[0];
    
    if(inValues == NULL || notDoubleVector1D(inValues))
    {
        PyErr_SetString(GETSTATE(self)->error, "inValues must be type double (numpy.float64) and 1D");
        return NULL;
    }
    size_t nInVals = inValues->dimensions[0];
    
    if( (nInX != nInY) | (nInX != nInVals) )
    {
        PyErr_SetString(GETSTATE(self)->error, "inX, inY and inValues must be the same length");
        return NULL;
    }
    
    if(xGrid == NULL || notDoubleVector2D(xGrid))
    {
        PyErr_SetString(GETSTATE(self)->error, "xGrid must be type double (numpy.float64) and 2D");
        return NULL;
    }
    size_t nXGridRows = xGrid->dimensions[0];
    size_t nXGridCols = xGrid->dimensions[1];
    
    if(yGrid == NULL || notDoubleVector2D(yGrid))
    {
        PyErr_SetString(GETSTATE(self)->error, "yGrid must be type double (numpy.float64) and 2D");
        return NULL;
    }
    size_t nYGridRows = yGrid->dimensions[0];
    size_t nYGridCols = yGrid->dimensions[1];
    
    printf ("X [%f, %f]\n", nXGridRows, nXGridCols);
    printf ("Y [%f, %f]\n", nYGridRows, nYGridCols);

    if( (nXGridRows != nYGridRows) | (nXGridCols != nYGridCols) )
    {
        PyErr_SetString(GETSTATE(self)->error, "xGrid and yGrid must be the same size.");
        return NULL;
    }
    
    double *inXVals = (double *) inX->data;
    double *inYVals = (double *) inY->data;
    double *inVals = (double *) inValues->data;
    double *xGridVals = (double*) xGrid->data;
    double *yGridVals = (double*) yGrid->data;
    
    for(int i = 0; i < nInX; ++i)
    {
        printf ("%d [%f, %f, %f]\n", i, inXVals[i], inYVals[i], inVals[i]);
    }
    
    
    
    
    
    
    

    printf ("Hello World.\n");
    
    /*
    PyObject *outGrid;
    
    outGrid = PyArray_NewLikeArray(xGrid, NPY_ANYORDER, NULL, 0);
    if (outGrid == NULL)
    {
        return NULL;
    }
    
    return outGrid;
     */
    Py_RETURN_NONE;
}


static PyMethodDef PyNNInterpMethods[] = {
    {"interpNNGrid", (PyCFunction)PyNNInterp_interpNNGrid, METH_VARARGS | METH_KEYWORDS,
        "interpNNGrid(x=numpy.array, y=numpy.array, values=numpy.array, xGrid=numpy.array, yGrid=numpy.array)\n"
        "Interpolates values  for x,y locations on a grid defined by xGrid and yGrid.\n"
        "Where:\n"
        "\n"
        "* x is a numpy array\n"
        "* y is a numpy array\n"
        "* values is a numpy array\n"
        "* xGrid is a numpy array\n"
        "* yGrid is a numpy array\n"
        "\n"},
        {NULL}        /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
static int PyNNInterp_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int PyNNInterp_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_PyNNInterp",
    NULL,
    sizeof(struct PyNNInterpState),
    PyNNInterpMethods,
    NULL,
    PyNNInterp_traverse,
    PyNNInterp_clear,
    NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_PyNNInterp(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_PyNNInterp(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *pModule = PyModule_Create(&moduledef);
#else
    PyObject *pModule = Py_InitModule("_PyNNInterp", PyNNInterpMethods);
#endif
    if( pModule == NULL )
        INITERROR;
    
    struct PyNNInterpState *state = GETSTATE(pModule);
    
    // Create and add our exception type
    state->error = PyErr_NewException("_PyNNInterp.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    
    
#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}

