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
  void setXMLWeights();
  unsigned int _h;
  unsigned int _d_in;
  unsigned int _d_out;
  unsigned int _n;

  FileName _weightsFile;
  std::vector<DenseMatrix<Real>> _weights;
  std::vector<DenseVector<Real>> _bias;

  MultiMooseEnum _layerActivationFunctionEnum;

};
