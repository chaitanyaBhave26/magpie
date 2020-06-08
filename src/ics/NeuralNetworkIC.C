/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/* MAGPIE - Mesoscale Atomistic Glue Program for Integrated Execution */
/*                                                                    */
/*            Copyright 2017 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/


#include "NeuralNetworkIC.h"
#include "MooseMesh.h"

registerMooseObject("MagpieApp", NeuralNetworkIC);

InputParameters
NeuralNetworkIC::validParams()
{
  InputParameters params = InitialCondition::validParams();
  params.addClassDescription("Initializes variable using a neural network UserObject");
  params.addRequiredParam<UserObjectName>(
      "NeuralNetwork_user_object",
      "Name of the neural network user object that evaluates the nodal value");
  params.addRequiredCoupledVar("InputVariables", "Names of the non-linear variables for inputting to the neural net");
  return params;
}

NeuralNetworkIC::NeuralNetworkIC(const InputParameters & parameters)
  : InitialCondition(parameters),
    _nn_obj(getUserObject<NeuralNetwork>("NeuralNetwork_user_object"))
{
  _depend_vars.insert(name());
  _n_inputs = coupledComponents("InputVariables");

  _input_vect.resize(_n_inputs);
  for (unsigned int i = 0; i < _n_inputs; ++i)
    _input_vect[i] = &coupledValue("InputVariables", i);
}

Real
NeuralNetworkIC::value(const Point & p)
{
  DenseVector<Real> _input_layer(_n_inputs);
  for (unsigned int i = 0; i < _n_inputs; ++i)
  {
    auto * temp = _input_vect[i];
    _input_layer(i) = temp[0][0];
  }
  return _nn_obj.eval(_input_layer, 0);
}

const std::set<std::string> &
NeuralNetworkIC::getRequestedItems()
{
  return _depend_vars;
}
