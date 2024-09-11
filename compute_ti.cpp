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
  vector_flag = 0;
  size_vector = 0;
  extvector = 0;

  

  pstyle = utils::strdup(arg[3]);
  delta = utils::numeric(FLERR, arg[4], false, lmp);
  typeA = utils::numeric(FLERR, arg[5], false, lmp);
  if (typeA > atom->ntypes) error->all(FLERR,"Illegal compute TI atom type {}",typeH);
  typeB = utils::numeric(FLERR, arg[6], false, lmp);
  if (typeB > atom->ntypes) error->all(FLERR,"Illegal compute TI atom type {}",typeB);
  typeC = utils::numeric(FLERR, arg[6], false, lmp);
  if (typeC > atom->ntypes) error->all(FLERR,"Illegal compute TI atom type {}",typeC);

  // allocate space for charge, force, energy, virial arrays

  f_orig = nullptr;
  q_orig = nullptr;
  peatom_orig = keatom_orig = nullptr;
  pvatom_orig = kvatom_orig = nullptr;

  fixgpu = nullptr;

  allocate_storage();
}

/* ---------------------------------------------------------------------- */
ComputeTI::~ComputeTI()
{
   deallocate_storage();
   memory->destroy(epsilon_init);
   delete [] pparam;
   delete [] pstyle;
}

/* ---------------------------------------------------------------------- */

void ComputeTI::init()
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
   
   pparam = new char[pair_params[pstyle].length()+1];
   std::strcpy(pparam,pair_params[pstyle].c_str());
   
}

/* ---------------------------------------------------------------------- */

void ComputeTI::compute_scaler()
{
   double uA, uB, du_dl;
   double lA = -delta;
   double lB = delta;
   allocate_storage();
   backup_restore_qfev<1>();      // backup charge, force, energy, virial array values
   modify_epsilon_q(lA);      //
   update_lmp(); // update the lammps force and virial values
   uA = compute_epair(); // I need to define my own version using compute pe/atom // HA is for the deprotonated state with lambda==0
   modify_epsilon_q(lB);
   uB = compute_epair();
   backup_restore_qfev<-1>();      // restore charge, force, energy, virial array values
   deallocate_storage();
   du_dl = (uB-uA)/(lB-lA);
   scaler = du_dl;
}

/* ---------------------------------------------------------------------- */

void ComputeTI::allocate_storage()
{
  /* It should be nmax since in the case that 
     the newton flag is on the force in the 
     ghost atoms also must be update and the 
     nmax contains the maximum number of nlocal 
     and nghost atoms.
  */
  int nmax = atom->nmax; 
  memory->create(f_orig, nmax, 3, "compute_TI:f_orig");
  memory->create(q_orig, nmax, "compute_TI:q_orig");
  memory->create(peatom_orig, nmax, "compute_TI:peatom_orig");
  memory->create(pvatom_orig, nmax, 6, "compute_TI:pvatom_orig");
  if (force->kspace) {
     memory->create(keatom_orig, nmax, "compute_TI:keatom_orig");
     memory->create(kvatom_orig, nmax, 6, "compute_TI:kvatom_orig");
  }
}

/* ---------------------------------------------------------------------- */

void ComputeTI::deallocate_storage()
{
  memory->destroy(q_orig);
  memory->destroy(f_orig);
  memory->destroy(peatom_orig);
  memory->destroy(pvatom_orig);
  memory->destroy(keatom_orig);
  memory->destroy(kvatom_orig);

  f_orig = nullptr;
  peatom_orig = keatom_orig = nullptr;
  pvatom_orig = kvatom_orig = nullptr;
}

/* ----------------------------------------------------------------------
   Forward-reverse copy function to be used in backup_restore_qfev()
   ---------------------------------------------------------------------- */

template  <int direction>
void ComputeTI::forward_reverse_copy(double &a,double &b)
{
   if (direction == 1) a = b;
   if (direction == -1) b = a;
}

template  <int direction>
void ComputeTI::forward_reverse_copy(double* a,double* b, int i)
{
   if (direction == 1) a[i] = b[i];
   if (direction == -1) b[i] = a[i];
}

template  <int direction>
void ComputeTI::forward_reverse_copy(double** a,double** b, int i, int j)
{
   if (direction == 1) a[i][j] = b[i][j];
   if (direction == -1) b[i][j] = a[i][j];
}

/* ----------------------------------------------------------------------
   backup and restore arrays with charge, force, energy, virial
   taken from src/FEP/compute_fep.cpp
   backup ==> direction == 1
   restore ==> direction == -1
------------------------------------------------------------------------- */

template <int direction>
void ComputeTI::backup_restore_qfev()
{
  int i;

  int nall = atom->nlocal + atom->nghost;
  int natom = atom->nlocal;
  if (force->newton || force->kspace->tip4pflag) natom += atom->nghost;

  double **f = atom->f;
  for (i = 0; i < natom; i++)
    for (int j = 0 ; j < 3; j++)
       forward_reverse_copy<direction>(f_orig,f,i,j);
  
  double *q = atom->q;
  for (int i = 0; i < natom; i++)
     forward_reverse_copy<direction>(q_orig,q,i);

  forward_reverse_copy<direction>(eng_vdwl_orig,force->pair->eng_vdwl);
  forward_reverse_copy<direction>(eng_coul_orig,force->pair->eng_coul);

  for (int i = 0; i < 6; i++)
	  forward_reverse_copy<direction>(pvirial_orig ,force->pair->virial, i);

  if (update->eflag_atom) {
    double *peatom = force->pair->eatom;
    for (i = 0; i < natom; i++) forward_reverse_copy<direction>(peatom_orig,peatom, i);
  }
  if (update->vflag_atom) {
    double **pvatom = force->pair->vatom;
    for (i = 0; i < natom; i++)
      for (int j = 0; j < 6; j++)
      forward_reverse_copy<direction>(pvatom_orig,pvatom,i,j);
  }

  if (force->kspace) {
     forward_reverse_copy<direction>(energy_orig,force->kspace->energy);
     for (int j = 0; j < 6; j++)
         forward_reverse_copy<direction>(kvirial_orig,force->kspace->virial,j);
	  
     if (update->eflag_atom) {
        double *keatom = force->kspace->eatom;
        for (i = 0; i < natom; i++) forward_reverse_copy<direction>(keatom_orig,keatom,i);
     }
     if (update->vflag_atom) {
        double **kvatom = force->kspace->vatom;
        for (i = 0; i < natom; i++) 
	  for (int j = 0; j < 6; j++)
             forward_reverse_copy<direction>(kvatom_orig,kvatom,i,j);
     }
  }
}

/* --------------------------------------------------------------

   -------------------------------------------------------------- */
   
void ComputeTI::modify_epsilon_q(double& scale)
{
  int nlocal = atom->nlocal;
  int * mask = atom->mask;
  int * type = atom->type;
  int ntypes = atom->ntypes;
  double * q = atom->q;

  if (scale < 0) scale = 0;
  if (scale > 1) scale = 1;

  // not sure about the range of these two loops
  for (int i = 0; i < ntypes + 1; i++)
	 for (int j = i; j < ntypes + 1; j++)
    {
	    if (type[i] == typeA && scale >= 0)
	    	epsilon[i][j] = epsilon_init[i][j] * scale;
       if (type[i] == typeB && scale <= 1)
         epsilon[i][j] = epsilon_init[i][j] * (1-scale);
    }
}

/* ----------------------------------------------------------------------
   modify force and kspace in lammps according
   ---------------------------------------------------------------------- */

void ComputeTI::update_lmp() {
   int eflag = 1;
   int vflag = 1;
   timer->stamp();
   if (force->pair && force->pair->compute_flag) {
     force->pair->compute(eflag, vflag);
     timer->stamp(Timer::PAIR);
   }
   if (force->kspace && force->kspace->compute_flag) {
     force->kspace->compute(eflag, vflag);
     timer->stamp(Timer::KSPACE);
   }

   // accumulate force/energy/virial from /gpu pair styles
   if (fixgpu) fixgpu->post_force(vflag);
}

/* --------------------------------------------------------------------- */

void ComputeTI::compute_q_total()
{
   double * q = atom->q;
   double nlocal = atom->nlocal;
   double q_local = 0.0;
   double q_total;
   double tolerance = 0.001;

   for (int i = 0; i <nlocal; i++)
       q_local += q[i];

    MPI_Allreduce(&q_local,&q_total,1,MPI_DOUBLE,MPI_SUM,world);

    if ((q_total >= tolerance || q_total <= -tolerance) && comm->me == 0)
    	error->warning(FLERR,"q_total in fix constant-pH is non-zero: {}",q_total);
}

/* --------------------------------------------------------------------- */

double ComputeTI::compute_epair()
{
   //if (update->eflag_global != update->ntimestep)
   //   error->all(FLERR,"Energy was not tallied on the needed timestep");

   int natoms = atom->natoms;

   double energy_local = 0.0;
   double energy = 0.0;
   if (force->pair) energy_local += (force->pair->eng_vdwl + force->pair->eng_coul);

   /* As the bond, angle, dihedral and improper energies 
      do not change with the espilon, we do not need to 
      include them in the energy. We are interested in 
      their difference afterall */

   MPI_Allreduce(&energy_local,&energy,1,MPI_DOUBLE,MPI_SUM,world);
   energy /= static_cast<double> (natoms); // To convert to kcal/mol the total energy must be devided by the number of atoms
   return energy;
}
