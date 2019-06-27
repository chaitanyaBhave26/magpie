/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/* MAGPIE - Mesoscale Atomistic Glue Program for Integrated Execution */
/*                                                                    */
/*            Copyright 2017 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MultiAppTransfer.h"

class MultiAppNeutronicsSpectrumTransfer;

template <>
InputParameters validParams<MultiAppNeutronicsSpectrumTransfer>();

/**
 * Transfer a neutronics probability density to a neutronics based pka generator.
 */
class MultiAppNeutronicsSpectrumTransfer : public MultiAppTransfer
{
public:
  MultiAppNeutronicsSpectrumTransfer(const InputParameters & parameters);

  virtual void execute();

protected:
  UserObjectName _pka_generator_name;
  UserObjectName _neutronics_pdf_name;
};

