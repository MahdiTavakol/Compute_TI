/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_ti.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "input.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "timer.h"
#include "update.h"
#include "variable.h"

using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

ComputeTI::ComputeTI(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
  if (narg < ??) error->all(FLERR, "Illegal number of arguments in compute ti");

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = ??;
  extvector = 0;

  const int ntypes = atom->ntypes;
  vector = new double[size_vector];
  

  dlambda = utils::numeric(FLERR, arg[3], false, lmp);



  // allocate space for charge, force, energy, virial arrays

  f_orig = nullptr;
  q_orig = nullptr;
  peatom_orig = keatom_orig = nullptr;
  pvatom_orig = kvatom_orig = nullptr;

  allocate_storage();

  fixgpu = nullptr;
}

/* ---------------------------------------------------------------------- */

void FixConstantPH::init()
{
   std::map<std::string, std::string> pair_params;
   
   pair_params["lj/cut/soft/omp"] = "lambda";
   pair_params["lj/cut/coul/cut/soft/gpu"] = "lambda";
   pair_params["lj/cut/coul/cut/soft/omp"] = "lambda";
   pair_params["lj/cut/coul/long/soft"] = "lambda";
   pair_params["lj/cut/coul/long/soft/gpu"] = "lambda";
   pair_params["lj/cut/coul/long/soft/omp"] = "lambda";
   pair_params["lj/cut/tip4p/long/soft"] = "lambda";
   pair_params["lj/cut/tip4p/long/soft/omp"] = "lambda";
   pair_params["lj/charmm/coul/long/soft"] = "lambda";
   pair_params["lj/charmm/coul/long/soft/omp"] = "lambda";
   pair_params["lj/class2/soft"] = "lambda";
   pair_params["lj/class2/coul/cut/soft"] = "lambda";
   pair_params["lj/class2/coul/long/soft"] = "lambda";
   pair_params["coul/cut/soft"] = "lambda";
   pair_params["coul/cut/soft/omp"] = "lambda";
   pair_params["coul/long/soft"] = "lambda";
   pair_params["coul/long/soft/omp"] = "lambda";
   pair_params["tip4p/long/soft"] = "lambda";
   pair_params["tip4p/long/soft/omp"] = "lambda";
   pair_params["morse/soft"] = "lambda";
   
   pair_params["lj/charmm/coul/charmm"] = "lj14_1";
   pair_params["lj/charmm/coul/charmm/gpu"] = "lj14_1";
   pair_params["lj/charmm/coul/charmm/intel"] = "lj14_1";
   pair_params["lj/charmm/coul/charmm/kk"] = "lj14_1";
   pair_params["lj/charmm/coul/charmm/omp"] = "lj14_1";
   pair_params["lj/charmm/coul/charmm/implicit"] = "lj14_1";
   pair_params["lj/charmm/coul/charmm/implicit/kk"] = "lj14_1";
   pair_params["lj/charmm/coul/charmm/implicit/omp"] = "lj14_1";
   pair_params["lj/charmm/coul/long"] = "lj14_1";
   pair_params["lj/charmm/coul/long/gpu"] = "lj14_1";
   pair_params["lj/charmm/coul/long/intel"] = "lj14_1";
   pair_params["lj/charmm/coul/long/kk"] = "lj14_1";
   pair_params["lj/charmm/coul/long/opt"] = "lj14_1";
   pair_params["lj/charmm/coul/long/omp"] = "lj14_1";
   pair_params["lj/charmm/coul/msm"] = "lj14_1";
   pair_params["lj/charmm/coul/msm/omp"] = "lj14_1";
   pair_params["lj/charmmfsw/coul/charmmfsh"] = "lj14_1";
   pair_params["lj/charmmfsw/coul/long"] = "lj14_1";
   pair_params["lj/charmmfsw/coul/long/kk"] = "lj14_1";

   if (pair_params.find(pstyle) == pair_params.end())
      error->all(FLERR,"The pair style {} is not currently supported in fix constant_pH",pstyle);
   
   pparam1 = new char[pair_params[pstyle].length()+1];
   std::strcpy(pparam1,pair_params[pstyle].c_str());
   
}
