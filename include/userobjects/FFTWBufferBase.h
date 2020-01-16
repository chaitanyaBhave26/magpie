/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/* MAGPIE - Mesoscale Atomistic Glue Program for Integrated Execution */
/*                                                                    */
/*            Copyright 2017 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#ifdef FFTW3_ENABLED

#pragma once

#include "FFTBufferBase.h"

#include "fftw3.h"

template <typename T>
class FFTWBufferBase;

/**
 * FFTW specific interleaved data buffer base class
 */
template <typename T>
class FFTWBufferBase : public FFTBufferBase<T>
{
public:
  FFTWBufferBase(const InputParameters & parameters);
  ~FFTWBufferBase();

  // transforms
  void forward() override;
  void backward() override;

protected:
  /// FFTW plans
  fftw_plan _forward_plan;
  fftw_plan _backward_plan;
};

#endif
