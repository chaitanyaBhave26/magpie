//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

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
  params.addRequiredParam<std::vector<NonlinearVariableName>>(
      "InputVariables", "Names of the non-linear variables for inputting to the neural net");
  return params;
}

NeuralNetworkIC::NeuralNetworkIC(const InputParameters & parameters)
  : InitialCondition(parameters),
    _nn_obj(getUserObject<NeuralNetwork>("NeuralNetwork_user_object")),
    _var_names(getParam<std::vector<NonlinearVariableName>>("InputVariables"))
{
  _depend_vars.insert(name());
  const std::set<std::string> temp = _nn_obj.getRequestedItems();
  _depend_vars.insert(temp.begin(), temp.end());
}

Real
NeuralNetworkIC::value(const Point & p)
{
  return _nn_obj.eval();
}

const std::set<std::string> &
NeuralNetworkIC::getRequestedItems()
{
  return _depend_vars;
}
