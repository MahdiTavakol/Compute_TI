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

#include "compute_thermo_integ.h"

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
/* A compute style to do thermodynamic integration written by Mahdi Tavakol (Oxford) mahditavakol90@gmail.com */

using namespace LAMMPS_NS;

enum { SINGLE = 1 << 0, DUAL = 1 << 1 };
enum {
    PAIR = 1 << 0,
    CHARGE = 1 << 1,
    BOTH = 1 << 2
};

/* ---------------------------------------------------------------------- */

ComputeThermoInteg::ComputeThermoInteg(LAMMPS* lmp, int narg, char** arg) : Compute(lmp, narg, arg)
{
    if (narg < 8) error->all(FLERR, "Illegal number of arguments in compute ti");

    scalar_flag = 0;
    vector_flag = 1;
    size_vector = 3;
    extvector = 0;

    vector = new double[3];

    parameter_list = 0;
    mode = 0;

    int iarg;
    if (strcmp(arg[3], "dual") == 0)
    {
        mode |= DUAL;
        typeA = utils::numeric(FLERR, arg[4], false, lmp);
        if (typeA > atom->ntypes) error->all(FLERR, "Illegal compute TI atom type {}", typeA);
        typeB = utils::numeric(FLERR, arg[5], false, lmp);
        if (typeB > atom->ntypes) error->all(FLERR, "Illegal compute TI atom type {}", typeB);
        iarg = 6;
    }
    else if (strcmp(arg[3], "single") == 0)
    {
        mode |= SINGLE;
        typeA = utils::numeric(FLERR, arg[4], false, lmp);
        if (typeA > atom->ntypes) error->all(FLERR, "Illegal compute TI atom type {}", typeA);
        iarg = 5;
    }
    else error->all(FLERR, "Unknown compute TI style {}", arg[3]);


    while (iarg < narg)
    {
        if (strcmp(arg[iarg], "pair") == 0)
        {
            parameter_list |= PAIR;
            pstyle = utils::strdup(arg[iarg + 1]);
            delta_p = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
            iarg += 3;
        }
        else if (strcmp(arg[iarg], "charge") == 0)
        {
            parameter_list |= CHARGE;
            delta_q = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            typeC = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
            if (typeC > atom->ntypes) error->all(FLERR, "Illegal compute TI atom type {}", typeC);
            iarg += 3;
        }
        else error->all(FLERR, "Unknown compute TI keyword {}", arg[iarg]);
    }


    // allocate space for charge, force, energy, virial arrays

    f_orig = nullptr;
    q_orig = nullptr;
    peatom_orig = keatom_orig = nullptr;
    pvatom_orig = kvatom_orig = nullptr;

    fixgpu = nullptr;

    nmax = atom->nmax;

    allocate_storage();
}

/* ---------------------------------------------------------------------- */
ComputeThermoInteg::~ComputeThermoInteg()
{
    deallocate_storage();
    memory->destroy(epsilon_init);
    delete[] pparam;
    delete[] pstyle;
    delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeThermoInteg::setup()
{
    if (parameter_list & PAIR)
    {
        pair = nullptr;
        if (lmp->suffix_enable)
            pair = force->pair_match(std::string(pstyle) + "/" + lmp->suffix, 1);
        if (pair == nullptr)
            pair = force->pair_match(pstyle, 1); // I need to define the pstyle variable
        void* ptr1 = pair->extract(pparam, pdim);
        if (ptr1 == nullptr)
            error->all(FLERR, "Compute TI pair style {} was not found", pstyle);
        if (pdim != 2)
            error->all(FLERR, "Pair style parameter {} is not compatible with compute TI", pparam);

        epsilon = (double**)ptr1;

        int ntypes = atom->ntypes;
        memory->create(epsilon_init, ntypes + 1, ntypes + 1, "compute_TI:epsilon_init");

        // I am not sure about the limits of these two loops, please double check them
        for (int i = 0; i < ntypes + 1; i++)
            for (int j = i; j < ntypes + 1; j++)
                epsilon_init[i][j] = epsilon[i][j];
    }
}

/* ---------------------------------------------------------------------- */

void ComputeThermoInteg::init()
{
    if (parameter_list & PAIR)
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
        pair_params["lj/charmm/coul/long"] = "epsilon";
        pair_params["lj/charmm/coul/long/gpu"] = "epsilon";
        pair_params["lj/charmm/coul/long/intel"] = "epsilon";
        pair_params["lj/charmm/coul/long/kk"] = "epsilon";
        pair_params["lj/charmm/coul/long/opt"] = "epsilon";
        pair_params["lj/charmm/coul/long/omp"] = "epsilon";
        pair_params["lj/charmm/coul/msm"] = "lj14_1";
        pair_params["lj/charmm/coul/msm/omp"] = "lj14_1";
        pair_params["lj/charmmfsw/coul/charmmfsh"] = "lj14_1";
        pair_params["lj/charmmfsw/coul/long"] = "lj14_1";
        pair_params["lj/charmmfsw/coul/long/kk"] = "lj14_1";

        if (pair_params.find(pstyle) == pair_params.end())
            error->all(FLERR, "The pair style {} is not currently supported in fix constant_pH", pstyle);

        pparam = new char[pair_params[pstyle].length() + 1];
        std::strcpy(pparam, pair_params[pstyle].c_str());
    }

}

/* ---------------------------------------------------------------------- */

void ComputeThermoInteg::compute_vector()
{
    vector[0] = 0.0;
    vector[1] = 0.0;
    vector[2] = 0.0;
   
    if (parameter_list & PAIR)
    {
        /* It should be compute_du<PAIR,mode>(delta_p); But that does not work for some reasons!*/
        if (mode & SINGLE)
            vector[0] = compute_du<PAIR, SINGLE>(delta_p,0.0);
        if (mode & DUAL)
            vector[0] = compute_du<PAIR, DUAL>(delta_p,0.0);
    }
    if (parameter_list & CHARGE)
    {
        if (mode & SINGLE)
            vector[1] = compute_du<CHARGE, SINGLE>(0.0,delta_q);
        if (mode & DUAL)
            vector[1] = compute_du<CHARGE, DUAL>(0.0,delta_q);
    }
    if (parameter_list & BOTH)
    {
        if (mode & SINGLE)
            vector[2] = compute_du<BOTH,SINGLE>(delta_p,delta_q)
        if (mode & DUAL)
            vector[2] = compute_du<BOTH,DUAL>(delta_p,delta_q);
    }
}

/* ---------------------------------------------------------------------- */

template <int parameter, int mode>
double ComputeThermoInteg::compute_du(double& delta_p, double& delta_q)
{
    double uA, uB, du_dl;
    double lA_p = -delta_p;
    double lA_q = -delta_q;
    double lB_p = delta_p;
    double lB_q = delta_q;
    /* check if there is enough allocated memory */
    if (nmax < atom->nmax)
    {
        nmax = atom->nmax;
        deallocate_storage();
        allocate_storage();
    }
    backup_restore_qfev<1>();      // backup charge, force, energy, virial array values
    modify_epsilon_q<parameter, mode>(lA_p,lA_q);      //
    update_lmp(); // update the lammps force and virial values
    uA = compute_epair(); // I need to define my own version using compute pe/atom // HA is for the deprotonated state with lambda==0
    modify_epsilon_q<parameter, mode>(lB_p,lB_q);
    update_lmp(); // update the lammps force and virial values
    uB = compute_epair();
    backup_restore_qfev<-1>();      // restore charge, force, energy, virial array values
    update_lmp(); // update the lammps force and virial values
    du_dl = (uB - uA) / (lB - lA);
    return du_dl;
}

/* ---------------------------------------------------------------------- */

void ComputeThermoInteg::allocate_storage()
{
    /* It should be nmax since in the case that
       the newton flag is on the force in the
       ghost atoms also must be update and the
       nmax contains the maximum number of nlocal
       and nghost atoms.
    */
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

void ComputeThermoInteg::deallocate_storage()
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
void ComputeThermoInteg::forward_reverse_copy(double& a, double& b)
{
    if (direction == 1) a = b;
    if (direction == -1) b = a;
}

template  <int direction>
void ComputeThermoInteg::forward_reverse_copy(double* a, double* b, int i)
{
    if (direction == 1) a[i] = b[i];
    if (direction == -1) b[i] = a[i];
}

template  <int direction>
void ComputeThermoInteg::forward_reverse_copy(double** a, double** b, int i, int j)
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
void ComputeThermoInteg::backup_restore_qfev()
{
    int i;

    int nall = atom->nlocal + atom->nghost;
    int natom = atom->nlocal;
    if (force->newton || force->kspace->tip4pflag) natom += atom->nghost;

    double** f = atom->f;
    for (i = 0; i < natom; i++)
        for (int j = 0; j < 3; j++)
            forward_reverse_copy<direction>(f_orig, f, i, j);

    double* q = atom->q;
    for (int i = 0; i < natom; i++)
        forward_reverse_copy<direction>(q_orig, q, i);

    forward_reverse_copy<direction>(eng_vdwl_orig, force->pair->eng_vdwl);
    forward_reverse_copy<direction>(eng_coul_orig, force->pair->eng_coul);

    for (int i = 0; i < 6; i++)
        forward_reverse_copy<direction>(pvirial_orig, force->pair->virial, i);

    if (update->eflag_atom) {
        double* peatom = force->pair->eatom;
        for (i = 0; i < natom; i++) forward_reverse_copy<direction>(peatom_orig, peatom, i);
    }
    if (update->vflag_atom) {
        double** pvatom = force->pair->vatom;
        for (i = 0; i < natom; i++)
            for (int j = 0; j < 6; j++)
                forward_reverse_copy<direction>(pvatom_orig, pvatom, i, j);
    }

    if (force->kspace) {
        forward_reverse_copy<direction>(energy_orig, force->kspace->energy);
        for (int j = 0; j < 6; j++)
            forward_reverse_copy<direction>(kvirial_orig, force->kspace->virial, j);

        if (update->eflag_atom) {
            double* keatom = force->kspace->eatom;
            for (i = 0; i < natom; i++) forward_reverse_copy<direction>(keatom_orig, keatom, i);
        }
        if (update->vflag_atom) {
            double** kvatom = force->kspace->vatom;
            for (i = 0; i < natom; i++)
                for (int j = 0; j < 6; j++)
                    forward_reverse_copy<direction>(kvatom_orig, kvatom, i, j);
        }
    }
}

/* --------------------------------------------------------------

   -------------------------------------------------------------- */
template <int parameter, int mode>
void ComputeThermoInteg::modify_epsilon_q(double& delta_p, double& delta_q)
{
    int nlocal = atom->nlocal;
    int* mask = atom->mask;
    int* type = atom->type;
    int ntypes = atom->ntypes;
    double* q = atom->q;



    // taking care of cases for which epsilon or lambda become negative
    if (parameter & PAIR)
    {
        int bad_i = 0;
        int bad_j = 0;
        bool modified_delta = false;
        if (delta_p < 0)
        {
            for (int i = typeA; i < ntypes + 1; i++)
                if (epsilon_init[typeA][i] < -delta_p)
                {
                    if (epsilon[i][i] == 0) continue;
                    bad_j = i;
                    modified_delta = true;
                    delta_p = -epsilon_init[typeA][i];
                }
            for (int i = 1; i < typeA; i++)
                if (epsilon_init[i][typeA] < -delta_p)
                {
                    if (epsilon[i][i] == 0) continue;
                    bad_i = i;
                    modified_delta = true;
                    delta_p = -epsilon_init[i][typeA];
                }
        }
        if (delta_p > 0 && mode == DUAL)
        {
            for (int i = typeB; i < ntypes + 1; i++)
                if (epsilon_init[typeB][i] < delta_p)
                {
                    if (epsilon[i][i] == 0) continue;
                    bad_j = i;
                    modified_delta = true;
                    delta_p = epsilon_init[typeA][i];
                }
            for (int i = 1; i < typeB; i++)
                if (epsilon_init[i][typeB] < delta_p)
                {
                    if (epsilon[i][i] == 0) continue;
                    bad_i = i;
                    modified_delta = true;
                    delta_p = epsilon_init[i][typeA];
                }
        }
        if (modified_delta) 
        {
           error->warning(FLERR,"The delta value in compute_TI has been modified to {} since it is less than epsilon({},{})", delta_p,bad_i,bad_j);
        }



        for (int i = 0; i < ntypes + 1; i++)
            for (int j = i; j < ntypes + 1; j++)
            {
                if (i == typeA || j == typeA)
                {
                    epsilon[i][j] = epsilon_init[i][j] + delta_p;
                }
                if (mode == DUAL)
                    if (i == typeB || j == typeB)
                        epsilon[i][j] = epsilon_init[i][j] - delta_p;
            }
        pair->reinit();
    }
    if (parameter & CHARGE )
    {
        double chargeC;
        selected.typeA = typeA;
        selected.typeB = typeB;
        selected.typeC = typeC;
        count_atoms(selected);


        if (selected.countA == 0) error->warning(FLERR, "Total number of atoms of type {} in compute ti is zero", typeA);
        if (selected.countB == 0 && (mode & DUAL)) error->warning(FLERR, "Total number of atoms of type {} in compute ti is zero", typeB);
        if (selected.countC == 0) error->all(FLERR, "Total number of atoms of type {} in compute ti is zero", typeC);
        chargeC = (selected.countA * delta + selected.countB * (-delta)) / selected.countC;

        for (int i = 0; i < nlocal; i++)
        {
            if (type[i] == typeA)
                q[i] += delta_q;
            if (mode == DUAL)
                if (type[i] == typeB)
                    q[i] -= delta_q;
            if (type[i] == typeC)
                q[i] = chargeC;
        }
    }
}

/* --------------------------------------------------------------------- */

void ComputeThermoInteg::count_atoms(selected_types& selected)
{
    int nlocal = atom->nlocal;
    double* q = atom->q;
    int* type = atom->type;

    int typeA = selected.typeA;
    int typeB = selected.typeB;
    int typeC = selected.typeC;
    int& countA = selected.countA;
    int& countB = selected.countB;
    int& countC = selected.countC;

    int* counts_local = new int[3];
    int* counts = new int[3];

    for (int i = 0; i < nlocal; i++)
    {
        if (type[i] == typeA) counts_local[0]++;
        if (type[i] == typeB) counts_local[1]++;
        if (type[i] == typeC) counts_local[2]++;
    }

    MPI_Allreduce(counts_local, counts, 3, MPI_INT, MPI_SUM, world);

    countA = counts[0];
    countB = counts[1];
    countC = counts[2];

}

/* ----------------------------------------------------------------------
   modify force and kspace in lammps according
   ---------------------------------------------------------------------- */

void ComputeThermoInteg::update_lmp() {
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

void ComputeThermoInteg::compute_q_total()
{
    double* q = atom->q;
    double nlocal = atom->nlocal;
    double q_local = 0.0;
    double q_total;
    double tolerance = 0.001;

    for (int i = 0; i < nlocal; i++)
        q_local += q[i];

    MPI_Allreduce(&q_local, &q_total, 1, MPI_DOUBLE, MPI_SUM, world);

    if ((q_total >= tolerance || q_total <= -tolerance) && comm->me == 0)
        error->warning(FLERR, "q_total in compute TI is non-zero: {}", q_total);
}

/* --------------------------------------------------------------------- */

double ComputeThermoInteg::compute_epair()
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

    MPI_Allreduce(&energy_local, &energy, 1, MPI_DOUBLE, MPI_SUM, world);
    energy /= static_cast<double> (natoms); // To convert to kcal/mol the total energy must be devided by the number of atoms
    return energy;
}
