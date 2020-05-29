/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/* MAGPIE - Mesoscale Atomistic Glue Program for Integrated Execution */
/*                                                                    */
/*            Copyright 2017 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "NeuralNetwork.h"
#include "MooseEnum.h"

registerMooseObject("MagpieApp", NeuralNetwork);

template <>
InputParameters
validParams<NeuralNetwork>()
{
  InputParameters params = NodalUserObject::validParams();
  params.addClassDescription("Reconstructs a neural network from a file and evaluates it");
  params.addRequiredParam<unsigned int>("H", "Number of neurons in each hidden layer");
  params.addRequiredParam<unsigned int>("N", "Number of hidden layers");
  params.addRequiredParam<unsigned int>("D_in", "Number of inputs to neural net");
  params.addRequiredParam<unsigned int>("D_out", "Number of outputs from neural net");
  params.addRequiredParam<FileName>("weights_file", "Name of the file with the neuron weights");
  MooseEnum activationFunctionEnum("SIGMOID SOFTSIGN TANH LINEAR", "SIGMOID");
  params.template addParam<MooseEnum>("activation_function",
                                      activationFunctionEnum,
                                      "Name of the hidden neuron activation function");
  params.addRequiredCoupledVar("variable", "Name of the variable this object operates on");
  return params;
}

NeuralNetwork::NeuralNetwork(const InputParameters & parameters)
  : NodalUserObject(parameters),
    _H(getParam<unsigned int>("H")),
    _N(getParam<unsigned int>("N")),
    _D_in(getParam<unsigned int>("D_in")),
    _D_out(getParam<unsigned int>("D_out")),
    _weights_file(getParam<FileName>("weights_file")),
    _activation_function(
        getParam<MooseEnum>("activation_function").template getEnum<ActivationFunction>()),
    MooseVariableInterface<Real>(this,
                                 false,
                                 "variable",
                                 Moose::VarKindType::VAR_ANY,
                                 Moose::VarFieldType::VAR_FIELD_STANDARD),
    _fe_vars(getCoupledMooseVars())
{

  // open the NN weights file
  setWeights();
  _inputs.resize(_D_in);
  _depend_vars.insert(name());
  for (unsigned int i = 0; i < _D_in; ++i)
  {
    auto temp = _fe_vars[i]->name();
    _depend_vars.insert(temp);
    _inputs[i] = &coupledValue("variable", i);
  }
}

void
NeuralNetwork::setWeights()
{
  std::ifstream ifile;
  std::string line;
  ifile.open(_weights_file);
  if (!ifile)
  {
    paramError("weights_file", "Unable to open file");
  }

  unsigned int line_no = 0;

  DenseMatrix<Real> _W_input(_H, _D_in);
  DenseMatrix<Real> _bias_input(1, _H);
  DenseMatrix<Real> _W_output(_D_out, _H);
  DenseMatrix<Real> _bias_output(1, _D_out);

  // Read input LINEAR neuron weights
  for (std::size_t i = 0; i < _H; i++)
  {
    for (std::size_t j = 0; j < _D_in; j++)
    {
      if (!(ifile >> _W_input(i, j)))
        mooseError("Error reading weights from file", _weights_file);
    }
  }
  for (std::size_t i = 0; i < _H; i++)
  {
    if (!(ifile >> _bias_input(0, i)))
      mooseError("Error reading input bias from file", _weights_file);
  }
  _weights.push_back(_W_input);
  _bias.push_back(_bias_input);

  // CHECK if hidden neuron type has weights
  switch (_activation_function)
  {
    case ActivationFunction::LINEAR:
      for (int n = 0; n < _N; ++n) // For each hidden layer
      {
        DenseMatrix<Real> _W_hidden(_H, _H);
        DenseMatrix<Real> _bias_hidden(1, _H);

        for (std::size_t i = 0; i < _H; i++)
        {
          for (std::size_t j = 0; j < _H; j++)
          {
            if (!(ifile >> _W_hidden(i, j)))
              mooseError("Error reading weights from file", _weights_file);
          }
        }
        for (std::size_t i = 0; i < _H; i++)
        {
          if (!(ifile >> _bias_hidden(0, i)))
            mooseError("Error reading input bias from file", _weights_file);
        }
        _weights.push_back(_W_hidden);
        _bias.push_back(_bias_hidden);
      }
  }

  // Read OUTPUT LINEAR neuron weights
  for (unsigned int i = 0; i < _D_out; i++)
  {
    for (unsigned int j = 0; j < _H; j++)
    {
      if (!(ifile >> _W_output(i, j)))
        mooseError("Error reading weights from file", _weights_file);
    }
  }
  for (unsigned int i = 0; i < _D_out; i++)
  {
    if (!(ifile >> _bias_output(0, i)))
      mooseError("Error reading input bias from file", _weights_file);
  }
  _weights.push_back(_W_output);
  _bias.push_back(_bias_output);
}

void NeuralNetwork::finalize(/* arguments */)
{ /* code */
}

void NeuralNetwork::execute(/* arguments */) {}

void NeuralNetwork::initialize(/* arguments */) {}
void
NeuralNetwork::threadJoin(const UserObject & /*y*/)
{
}

const std::set<std::string> &
NeuralNetwork::getRequestedItems() const
{
  return _depend_vars;
}

Real
NeuralNetwork::eval() const
{

  DenseMatrix<Real> input(1, _D_in);
  DenseMatrix<Real> feed_forward(1, _H);

  for (std::size_t i = 0; i < _D_in; ++i)
  {
    auto * temp = _inputs[i];
    input(0, i) = temp[0][0];
  }

  // Feed forward input linear neurons
  input.right_multiply_transpose(_weights[0]);
  input.add(1, _bias[0]);

  // Feed forward hidden neurons
  for (int n = 0; n < _N; ++n)
  {
    switch (_activation_function)
    {
      case ActivationFunction::SIGMOID:
      {
        for (std::size_t i = 0; i < input.m(); ++i)
        {
          for (std::size_t j = 0; j < input.n(); ++j)
          {
            Real temp = 1 / (1 + std::exp(-1 * input(i, j)));
            input(i, j) = temp;
          }
        }
      }
      case ActivationFunction::TANH:
      {
        for (std::size_t i = 0; i < input.m(); ++i)
        {
          for (std::size_t j = 0; j < input.n(); ++j)
          {
            Real temp = std::tanh(input(i, j));
            input(i, j) = temp;
          }
        }
      }
    }

    // feed forward output LINEAR layer
    auto i = _weights.size() - 1;
    input.right_multiply_transpose(_weights[i]);
    input.add(1, _bias[i]);
    return input(0, 0);
  }
