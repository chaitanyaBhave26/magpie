//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "NeuralNetwork.h"
#include "MooseEnum.h"

registerMooseObject("MagpieApp", NeuralNetwork);

template<>
InputParameters
validParams<NeuralNetwork>()
{
  InputParameters params = NodalUserObject::validParams();
  params.addClassDescription("Reconstructs a neural network from a file and evaluates it");
  params.addRequiredParam<unsigned int>("H","Number of neurons in each hidden layer");
  params.addRequiredParam<unsigned int>("N","Number of hidden layers");
  params.addRequiredParam<unsigned int>("D_in","Number of inputs to neural net");
  params.addRequiredParam<unsigned int>("D_out","Number of outputs from neural net");
  params.addRequiredParam<FileName>("weights_file","Name of the file with the neuron weights");
  MooseEnum activationFunctionEnum("SIGMOID SOFTSIGN TANH LINEAR","SIGMOID");
  params.template addParam<MooseEnum>("activation_function",activationFunctionEnum,"Name of the hidden neuron activation function");
  params.addRequiredParam<std::vector <NonlinearVariableName>>("variables","List of non-linear variables to be used as input");
  params.addRequiredCoupledVar("variable","Name of the variable this object operates on");
  params.addParam<std::vector<std::string>>("IC_dependencies","List of ICs that need to be executed before this UO");
  return params;
}

NeuralNetwork::NeuralNetwork(const InputParameters & parameters)
  : NodalUserObject(parameters),
    _H(getParam<unsigned int>("H")),
    _N(getParam<unsigned int>("N")),
    _D_in(getParam<unsigned int>("D_in")),
    _D_out(getParam<unsigned int>("D_out")),
    _weights_file(getParam<FileName>("weights_file")),
    _activation_function(getParam<MooseEnum>("activation_function").template getEnum<ActivationFunction>() ),
    _variables(getParam<std::vector <NonlinearVariableName>>("variables")  ),
    _ic_dependencies(getParam<std::vector<std::string>>("IC_dependencies")),
    MooseVariableInterface<Real>(this,
                               false,
                               "variables",
                               Moose::VarKindType::VAR_ANY,
                               Moose::VarFieldType::VAR_FIELD_STANDARD),
    _var(*mooseVariable()),
    _u(_var.dofValues()),
    // _var_vals(coupledValue("variable",1)),
    _fe_vars(getCoupledMooseVars())
{

  //open the NN weights file
  setWeights();
  _inputs.resize(_D_in);
  _depend_vars.insert(name());
  for(unsigned int i =0; i < _fe_vars.size(); ++i)
    {
      auto temp = _fe_vars[i]->name();
      _depend_vars.insert(temp);
      _inputs[i] = &coupledValue("variable",i) ;
    }
  for(unsigned int i = 0; i < _ic_dependencies.size(); ++i)
    {
      _depend_vars.insert(_ic_dependencies[i]);
    }

}

void NeuralNetwork::setWeights()
  {
    std::ifstream ifile;
    std::string line;
    ifile.open(_weights_file);
    if(!ifile)
      {
        paramError("weights_file","Unable to open file");
      }

    unsigned int line_no = 0;

    DenseMatrix<Real> _W_input(_H,_D_in);
    DenseMatrix<Real> _bias_input(1,_H);
    DenseMatrix<Real> _W_output(_D_out,_H);
    DenseMatrix<Real> _bias_output(1,_D_out);

    //Read input LINEAR neuron weights
    for (std::size_t i =0; i < _H; i++)
      {
        for (std::size_t j=0; j < _D_in; j++)
          {
            if (!(ifile >> _W_input(i,j) ) )
                mooseError("Error reading weights from file",_weights_file);

          }
      }
    for (std::size_t i =0; i < _H; i++)
      {
        if (!(ifile >> _bias_input(0,i) ))
          mooseError("Error reading input bias from file",_weights_file);
      }
    _weights.push_back(_W_input);
    _bias.push_back(_bias_input);

    //CHECK if hidden neuron type has weights
    switch (_activation_function)
    {
      case ActivationFunction::LINEAR:
        for (int n =0; n < _N; ++n)         //For each hidden layer
          {
            DenseMatrix<Real> _W_hidden(_H,_H);
            DenseMatrix<Real> _bias_hidden(1,_H);

            for (std::size_t i =0; i < _H; i++)
              {
                for (std::size_t j=0; j < _H; j++)
                  {
                    if (!(ifile >> _W_hidden(i,j) ) )
                      mooseError("Error reading weights from file",_weights_file);

                  }
              }
            for (std::size_t i =0; i < _H; i++)
              {
                if (!(ifile >> _bias_hidden(0,i) ))
                  mooseError("Error reading input bias from file",_weights_file);
              }
            _weights.push_back(_W_hidden);
            _bias.push_back(_bias_hidden);

          }


    }

    //Read OUTPUT LINEAR neuron weights
    for (unsigned int i =0; i < _D_out; i++)
      {
        for (unsigned int j=0; j < _H; j++)
          {
            if (!(ifile >> _W_output(i,j) ) )
              mooseError("Error reading weights from file",_weights_file);

          }
      }
    for (unsigned int i =0; i < _D_out; i++)
      {
        if (!(ifile >> _bias_output(0,i) ))
          mooseError("Error reading input bias from file",_weights_file);
      }
    _weights.push_back(_W_output);
    _bias.push_back(_bias_output);

  }


void
NeuralNetwork::finalize(/* arguments */) {
  /* code */
}

void
NeuralNetwork::execute(/* arguments */) {

}

void
NeuralNetwork::initialize(/* arguments */) {
}
void
NeuralNetwork::threadJoin(const UserObject & /*y*/)
{
}

const std::set<std::string> &
NeuralNetwork::getRequestedItems()  const
{
  return _depend_vars;
}

Real
NeuralNetwork::eval( ) const
{

  DenseMatrix<Real> input(1,_D_in);
  DenseMatrix<Real> feed_forward(1,_H);

  for (std::size_t i =0; i<_D_in; ++i)
    {
      auto *temp = _inputs[i];
      input(0,i)  = temp[0][0];
    }

//Feed forward input linear neurons
  std::cout << "Input state: ";
  std::cout << input.m() << ", " << input.n() << "\n " << _weights[0].m() << ", " << _weights[0].n();

  input.left_multiply_transpose(_weights[0]);
  input.add(1,_bias[0]);

  std::cout << "\n After input feed forward: ";
  std::cout << input.m() << ", " << input.n();


//Feed forward hidden neurons
  for (int n=0; n < _N; ++n)
  {
    switch (_activation_function)
    {
      case ActivationFunction::SIGMOID:
        for (std::size_t i =0; i < input.m(); ++i)
          {
            for (std::size_t j =0; j < input.n(); ++j)
              {
                Real temp = 1/(1 + std::exp(-1*input(i,j)));
                input(i,j)=temp;
              }
          }
          std::cout << "\n After SIGMOID feed forward: ";
          std::cout << input.m() << ", " << input.n();
     }
  }


  //feed forward output LINEAR layer
  auto i = _weights.size() -1;
  std::cout << input.m() << ", " << input.n() << "\n " << _weights[i].m() << ", " << _weights[i].n();
  input.left_multiply_transpose(_weights[i]);
  input.add(1,_bias[i]);
  return input(0,0);
}

// void
// NeuralNetwork::ApplyLinearInput( std::vector<Real> & input,std::vector<Real> & output) const
//   {
//     // std::cout << "Inputs " << input.size() << "\n";
//     output.resize(_H);
//     // Output = Input*Weights
//     for (int i=0; i< _H; ++i)
//       {
//         Real temp = 0;
//         for (int j = 0; j< _D_in; ++j)
//           {
//             temp+=input[j]*_weights[0][i][j];
//           }
//         output[i] = temp + _bias[0][i];
//
//       }
//
//   }
//
// void
// NeuralNetwork::ApplyLinearOutput( std::vector<Real> & input,Real & output) const
//   {
//     // std::cout << "Inputs " << input.size() << "\n";
//     if (! (_D_out == 1) )
//       {
//         mooseError("No implementation for an NN with more than 1 outputs yet");
//       }
//     // Output = Input*Weights
//     for (int i=0; i< _H; ++i)
//       {
//         output+=input[i]*_weights[1][0][i];
//
//       }
//       output += _bias[1][0];
//       if (output >= 1.0)
//         {
//           output = 1.0 - 1e-3;
//         }
//       else if (output <= 0.0)
//         {
//           output = 1e-3;
//         }
//   }
