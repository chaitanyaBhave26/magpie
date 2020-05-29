/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/* MAGPIE - Mesoscale Atomistic Glue Program for Integrated Execution */
/*                                                                    */
/*            Copyright 2017 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once
#include "NodalUserObject.h"
#include "MooseVariableInterface.h"

// forward declarations
class NeuralNetwork;

template <>
InputParameters validParams<NodalUserObject>();

class NeuralNetwork : public NodalUserObject, public MooseVariableInterface<Real>
{
public:
  static InputParameters validParams();

  NeuralNetwork(const InputParameters & parameters);
  virtual void initialize() override;
  virtual void execute() override;
  virtual void finalize() override;
  virtual void threadJoin(const UserObject & y) override;
  const std::set<std::string> & getRequestedItems() const;
  Real eval() const;

protected:
  void setWeights();

  unsigned int _H;
  unsigned int _D_in;
  unsigned int _D_out;
  unsigned int _N;

  FileName _weights_file;
  std::vector<NonlinearVariableName> _variables;
  MooseVariable & _var;
  const VariableValue & _u;
  std::vector<const VariableValue *> _inputs;
  std::vector<MooseVariableFEBase *> _fe_vars;

  std::vector<DenseMatrix<Real>> _weights;
  std::vector<DenseMatrix<Real>> _bias;
  std::set<std::string> _depend_vars;

  enum class ActivationFunction
  {
    SIGMOID,
    SOFTSIGN,
    TANH,
    LINEAR
  } _activation_function;
};
