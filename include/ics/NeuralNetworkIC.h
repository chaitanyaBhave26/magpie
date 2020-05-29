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

  // virtual void initialSetup();

protected:
  const NeuralNetwork & _nn_obj;
  std::vector<NonlinearVariableName> _var_names;
  // MooseMesh & _mesh;

};
