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

//forward declarations
class NeuralNetwork;

template <>
InputParameters validParams<NodalUserObject>();

class NeuralNetwork: public NodalUserObject,
                     public MooseVariableInterface<Real>
{
public:
  static InputParameters validParams();

  NeuralNetwork(const InputParameters & parameters);
  virtual void initialize() override;
  virtual void execute() override;
  virtual void finalize() override;
  virtual void threadJoin(const UserObject & y) override;
  // void NN_eval();
  const std::set<std::string> & getRequestedItems() const;
  // const std::set<std::string> & getRequestedItems() override;
  // virtual const std::set<std::string> getRequestedItems() override;
  // virtual Real spatialValue(const Point & p) const override {return calc_spatial_value(p);}
  Real eval() const;
  // std::set<std::string> _supplied_vars;

  // const Point & p;

protected:
  ///@{ MooseMesh Variables
  // MooseMesh & _mesh;
  // NonlinearSystemBase & _nl;
  ///@}
  // SystemBase & _sys;
  // virtual unsigned int map_MOOSE2Ext(const Node & MOOSEnode) const;
  void setWeights();

  unsigned int _H;
  unsigned int _D_in;
  unsigned int _D_out;
  unsigned int _N;

  FileName _weights_file;
  // std::string _activation_function;
  std::vector< NonlinearVariableName> _variables;
  // MooseVariableField & _var;
  MooseVariable & _var;
  const VariableValue & _u;
  // const VariableValue &  _var_vals;
  std::vector< const VariableValue *> _inputs;
  // const std::set<std::string> _var_names;
  std::vector<MooseVariableFEBase *> _fe_vars;

  std::vector<DenseMatrix <Real>> _weights;
  std::vector< DenseMatrix <Real>> _bias;
  std::set<std::string> _depend_vars;
  std::vector<std::string> _ic_dependencies;
  // std::vector<>

  enum class ActivationFunction
  {
    SIGMOID,
    SOFTSIGN,
    TANH,
    LINEAR
  } _activation_function;

// private:
//   void ApplyLinearInput(std::vector<Real> & input,std::vector<Real> & output) const;
//   void ApplyLinearOutput( std::vector<Real> & input,Real & output) const;
  // std::vector<Real> _bias_input;
  // std::vector<Real> _bias_output;
  // std::vect<
  // const Point & p;
};
