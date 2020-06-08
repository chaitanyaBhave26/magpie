/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/* MAGPIE - Mesoscale Atomistic Glue Program for Integrated Execution */
/*                                                                    */
/*            Copyright 2017 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once
#include "GeneralUserObject.h"
#include "MooseVariableInterface.h"

// forward declarations
class NeuralNetwork;

template <>
InputParameters validParams<GeneralUserObject>();

class NeuralNetwork : public GeneralUserObject
{
public:
  static InputParameters validParams();

  NeuralNetwork(const InputParameters & parameters);
  virtual void initialize() override {}
  virtual void execute() override {}
  virtual void finalize() override {}
  virtual void threadJoin(const UserObject & y) override {}
  Real eval(DenseVector<Real> & input, std::size_t op_idx) const;

protected:
  void setWeights();

  unsigned int _H;
  unsigned int _D_in;
  unsigned int _D_out;
  unsigned int _N;

  FileName _weights_file;
  std::vector<DenseMatrix<Real>> _weights;
  std::vector<DenseVector<Real>> _bias;

  enum class ActivationFunction
  {
    SIGMOID,
    SOFTSIGN,
    TANH,
    LINEAR
  } _activation_function;

};
