/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(ti,ComputeTI);
// clang-format on
#else

#ifndef COMPUTE_TI_H
#define COMPUTE_TI_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeTI : public Compute {
 public:
  ComputeTI(class LAMMPS *, int, char **);
  ~ComputeTI() override;
  void init() override;
  void compute_vector() override;

 private:

  // Pair style parameters
  char * pstyle, * pparam1;
  Pair * pair1;
  int pdim1;

  void compute_us();
  void allocate_storage();
  void deallocate_storage();
  template  <int direction>
  void forward_reverse_copy(double &,double &);
  template  <int direction>
  void forward_reverse_copy(double* ,double* , int );
  template  <int direction>
  void forward_reverse_copy(double** ,double** , int , int );
  template <int direction>
  void backup_restore_qfev();
  void modify_epsilon_q();
  void update_lmp();
  void compute_q_total();
  void compute_epair();
};

}    // namespace LAMMPS_NS

#endif
#endif
