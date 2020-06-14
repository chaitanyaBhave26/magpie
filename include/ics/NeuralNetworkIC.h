/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/* MAGPIE - Mesoscale Atomistic Glue Program for Integrated Execution */
/*                                                                    */
/*            Copyright 2017 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "InitialCondition.h"
#include "NeuralNetwork.h"

class NeuralNetworkIC : public InitialCondition
{
public:
  static InputParameters validParams();

  NeuralNetworkIC(const InputParameters & parameters);

  virtual Real value(const Point & p);
  virtual const std::set<std::string> & getRequestedItems() override;

protected:
  const NeuralNetwork & _nn_obj;
  std::vector<const VariableValue *> _input_vect;
  unsigned int _n_inputs;
  std::size_t _op_id;
};
