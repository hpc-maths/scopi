#include <random>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>
using namespace xt::placeholders;

#include <scopi/objects/types/segment.hpp>
#include <scopi/objects/types/sphere.hpp> //to create spheres
#include <scopi/solver.hpp>               //to create the solver and run it

// To write a json file
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <typeinfo> // For typeid

// Distance function: function declaration
double computeDistance(double x1, double y1, double x2, double y2);

// Function to compute the Euclidean distance between two points
template <std::size_t dim>
double dist(const scopi::type::position_t<dim>& q1, const scopi::type::position_t<dim>& q2)
{
    return std::sqrt(std::pow(q2[0] - q1[0], 2) + std::pow(q2[1] - q1[1], 2));
}

template <std::size_t dim>
scopi::type::velocity_t<dim>
spont_u_1(const scopi::type::position_t<dim>& q1, double s, double param, double l_salle, double L_salle, double bord, double R)
{
    double l_porte = param;

    double l_mur1 = (l_salle - l_porte) / 2;
    double l_mur2 = l_mur1;

    double x = q1[0];
    double y = q1[1];

    // Check boundary conditions
    if (y >= (bord + L_salle) || (x >= (bord + l_mur1 + R) && x <= (bord + l_salle - l_mur2 - R)))
    {
        return {0, s};
    }
    scopi::type::position_t<dim> q2; // Initialize q2 as a vector of 2 elements

    if (x < (bord + l_mur1 + R))
    {
        q2 = {l_mur1 + R + bord, bord + L_salle - R};
    }
    else if (x > (bord + l_salle - l_mur2 - R))
    {
        q2 = {bord + l_salle - l_mur2 - R, bord + L_salle - R};
    }

    return (q2 - q1) * s / dist(q1, q2);
}

int main(int argc, char** argv)
{
    constexpr std::size_t dim = 2;
    double dt                 = .002; // 0.001
    const double max_radius   = 0.05;
    std::size_t total_it      = 1;  // 51 total time iterations. 6
    std::size_t n_parts       = 10; // 20

    scopi::initialize("spheres passing between two segments"); // just adds a title to your command line option

    auto& app = scopi::get_app();
    app.add_option("--nparts", n_parts, "Number of particles")->capture_default_str();
    app.add_option("--nite", total_it, "Number of iterations")->capture_default_str();
    app.add_option("--dt", dt, "Time step")->capture_default_str();

    // Parametric test:train
    std::size_t ntrain    = 10;
    std::size_t nsim      = ntrain;
    std::string flag_test = "train";
    unsigned int seed     = 1;

    // Parametric test:valid
    //  std::size_t nvalid = 1;
    //  std::size_t nsim=nvalid;
    //  std::string flag_test="valid";
    //  unsigned int seed = 54321;

    std::default_random_engine generator;
    generator.seed(seed);

    // Crée un objet JSON vide
    nlohmann::json json_obj;

    // Save the setting of the simulation
    json_obj["Tmax"].push_back(total_it);
    json_obj["Na"].push_back(n_parts);
    json_obj["deltat"].push_back(dt);
    json_obj["R"].push_back(max_radius);

    // double length=1;//0.4
    double l_salle = 2.25; // it is constant
    double l_mur   = 0.05;
    double ymur    = 1.0;
    double start   = 0.;

    // Sauvegarder aussi les autres parametres geometriques
    json_obj["l_mur"].push_back(l_mur);
    json_obj["start"].push_back(start);
    json_obj["ymur"].push_back(ymur);
    json_obj["l_salle"].push_back(l_salle);

    for (std::size_t itrain = 0; itrain < nsim; ++itrain)
    {
        scopi::scopi_container<dim> particles; // container of all the particles: why has the scopi container hpp not been included?
        scopi::ScopiSolver<dim> solver(particles);
        SCOPI_PARSE(argc, argv); // allows to get access to internal options

        // Distribution for lexit
        std::uniform_real_distribution<double> distrib_lexit(0.16, 0.24); // lexit reference=0.2
        auto lexit = distrib_lexit(generator);                            // varying parameter: exit width

        // Distribution for spontaneous velocity
        double sbar = 5.0;
        std::uniform_real_distribution<double> distrib_s(sbar - 0.1 * sbar, sbar + 0.1 * sbar);
        auto s_vel = distrib_s(generator); // For now, the spontaneous velocity is the same for all the particles. It only varies for
                                           // different geometric configurations

        double length = 0.5 * (l_salle - lexit);

        scopi::segment<dim> seg1(scopi::type::position_t<dim>{start, ymur}, scopi::type::position_t<dim>{length, ymur});
        scopi::segment<dim> seg2(scopi::type::position_t<dim>{length + lexit, ymur},
                                 scopi::type::position_t<dim>{length + length + lexit, ymur});
        scopi::segment<dim> seg3(scopi::type::position_t<dim>{length, ymur}, scopi::type::position_t<dim>{length, ymur + l_mur});
        scopi::segment<dim> seg4(scopi::type::position_t<dim>{length + lexit, ymur},
                                 scopi::type::position_t<dim>{length + lexit, ymur + l_mur});
        scopi::segment<dim> seg5(scopi::type::position_t<dim>{start, ymur + l_mur}, scopi::type::position_t<dim>{length, ymur + l_mur});
        scopi::segment<dim> seg6(scopi::type::position_t<dim>{length + lexit, ymur + l_mur},
                                 scopi::type::position_t<dim>{length + length + lexit, ymur + l_mur});
        scopi::segment<dim> seg7(scopi::type::position_t<dim>{start, ymur}, scopi::type::position_t<dim>{start, ymur + l_mur});
        scopi::segment<dim> seg8(scopi::type::position_t<dim>{length + length + lexit, ymur},
                                 scopi::type::position_t<dim>{length + length + lexit, ymur + l_mur});

        particles.push_back(seg1, scopi::property<dim>().deactivate()); // these objects are only obstacles
        particles.push_back(seg2, scopi::property<dim>().deactivate());
        particles.push_back(seg3, scopi::property<dim>().deactivate());
        particles.push_back(seg4, scopi::property<dim>().deactivate());
        particles.push_back(seg5, scopi::property<dim>().deactivate());
        particles.push_back(seg6, scopi::property<dim>().deactivate());
        particles.push_back(seg7, scopi::property<dim>().deactivate());
        particles.push_back(seg8, scopi::property<dim>().deactivate());

        std::uniform_real_distribution<double> distrib_x(0.5, 2 * length + lexit - 0.5);
        std::uniform_real_distribution<double> distrib_y(0.5, ymur - 0.1); // 0.2 min
        // std::uniform_real_distribution<double> distrib_r(0.01, max_radius);//r is constant for me

        for (std::size_t i = 0; i < n_parts; ++i)
        {
            auto x      = distrib_x(generator);
            auto y      = distrib_y(generator);
            auto radius = max_radius; // distrib_r(generator);//0.05
            // double dist_factor=computeDistance(start+length+0.5*lexit,2, x,y);

            // Call function spont_u: it should be the same used in Python
            // std::vector<double> q1{x,y};
            // auto prop   = scopi::property<dim>()
            //                 .desired_velocity(q1, s_vel,lexit,  l_salle,  ymur, 0.0, radius)
            //                 .mass(1.)
            //                 .moment_inertia(0.1);

            // particles.push_back(scopi::sphere<dim>(
            //                         {
            //                             {x, y}
            // },
            //                         radius),
            //                     prop);

            auto prop = scopi::property<dim>()
                            //                 .desired_velocity({
                            //                     {(start+length+0.5*lexit - x)*s_vel/dist_factor, (2 - y)*s_vel/dist_factor}// ymur+l_mur
                            // })
                            .desired_velocity(spont_u_1(scopi::type::position_t<dim>{x, y}, s_vel, lexit, l_salle, ymur, 0.0, radius))
                            .mass(1.)
                            .moment_inertia(0.1);

            particles.push_back(scopi::sphere<dim>(
                                    {
                                        {x, y}
            },
                                    radius),
                                prop);
        }

        auto params                                 = solver.get_params();
        params.contact_method_params.dmax           = 20000; // this is the parameter to modify to take all the contacts into account 2 * dt
        params.contact_method_params.kd_tree_radius = 40000; // 4 * max_radius
        params.solver_params.write_velocity         = true;
        params.solver_params.write_lagrange_multiplier = true;
        if (flag_test == "train")
        {
            params.solver_params.path = "test_dir";
        }
        else if (flag_test == "valid")
        {
            params.solver_params.path = "test_dir_valid";
        }
        params.solver_params.filename = "test" + std::to_string(itrain); // file in which write the solutions// parametric test: +
                                                                         // std::to_string(itrain)

        // for (std::size_t it = 0; it < total_it; ++it)
        // {
        //     solver.run(dt, it + 1, it);
        //     for (std::size_t ip = 8; ip < particles.size(); ++ip) // Particles with indices 0 to 7 are excluded because they represent
        //                                                           // inactive particles =obstacles.
        //     {
        //         particles.vd()[ip] = spont_u_1(particles.pos()[ip], s_vel, lexit, l_salle, ymur, 0.0, max_radius);
        //     }
        // }

        solver.run(dt, 100);

        // auto contacts = solver.compute_contacts();

        // std::size_t index = 0;
        // for (auto& c: contacts)
        // {
        //     if (c.i < 8)
        //     {
        //         index++;
        //     }
        //     else
        //     {
        //         break;
        //     }
        // }
        // //std::cout << index << std::endl;

        // detail::sort_contacts_over_j(contacts, index);//chiama la template function a inizio file

        // using contacts_t = std::decay_t<decltype(contacts)>;//contacts_t is the type of contacts
        // contacts_t contacts2rom;//new vector to store
        // int j = -1;
        // for (std::size_t ic=0; ic <index; ++ic)//iterates through the sorted contacts
        // {
        //     auto& c = contacts[ic];
        //     if (c.j != j)
        //     {
        //         contacts2rom.push_back(c);
        //         j = c.j;
        //     }
        //     while (j == contacts[ic+1].j)
        //     {
        //         if (contacts2rom.back().dij > contacts[ic+1].dij)
        //         {
        //             contacts2rom.back() = contacts[ic+1];
        //         }
        //         ic++;
        //     }
        //     std::swap(contacts2rom.back().pi, contacts2rom.back().pj);
        //     contacts2rom.back().nij *= -1;
        //     contacts2rom.back().i = contacts2rom.back().j - 8;
        //     contacts2rom.back().j = std::numeric_limits<std::size_t>::max();//Sets i to an invalid state
        //     (std::numeric_limits<std::size_t>::max()), indicating it is no longer relevant.
        // }

        // //std::cout << "contact number between particles: " << contacts.size() - index << " (contacts size: " << contacts.size() << "
        // index: " << index << ")" << std::endl; for (std::size_t ic=index; ic <contacts.size(); ++ic)
        // {
        //     auto& c = contacts[ic];
        //     contacts2rom.push_back(c);
        //     contacts2rom.back().j -= 8;
        //     contacts2rom.back().i -= 8;
        // }

        // scopi::sort_contacts(contacts2rom);
        // //std::cout << "New contact size " << contacts2rom.size() << std::endl;
        // BMatrix<contacts_t> B(dt, contacts2rom);

        // std::string npy_filename_Vr = "/Users/gsambata/scopi/build/files2SCoPI/RB_v.npy";
        // std::string npy_filename_Wr = "/Users/gsambata/scopi/build/files2SCoPI/RB_lambda.npy";

        // auto W = xt::load_npy<float>(npy_filename_Wr);
        // auto WT = xt::eval(xt::transpose(W));
        // auto V = xt::load_npy<float>(npy_filename_Vr);
        // auto VT = xt::eval(xt::transpose(V));
        // //These are 2dimensional! (compute shape())

        // auto BV = B.mat_mult(V);//Multiplies the "contact matrix" (implicitly defined by m_contacts) with the RB matrix Vr
        // xt::xtensor<double, 2>  Bhat = xt::linalg::dot(WT, BV); // W^TBV
        // //std::cout << Bhat << std::endl;
        // json_obj["B"].push_back(B.buildB(particles.nb_active()));
        // json_obj["BV"].push_back(BV);
        // json_obj["Bhat"].push_back(Bhat);

        // // Compute Qhat
        // xt::xtensor<double, 2> Qhat = xt::linalg::dot(Bhat, xt::transpose(Bhat));
        // json_obj["Qhat"].push_back(Qhat);

        // //Compute Chat
        // //Compute spontaneous velocities vector
        // //xt::xtensor<double> U({static_cast<double>(n_parts)});
        // //xt::xtensor<scopi::type::velocity_t<2>,10 > U;
        // //for(std::size_t ip = 8; ip < particles.size(); ++ip){
        //     //U[ip]=spont_u_1(particles.pos()[ip], s_vel,lexit, l_salle, ymur, 0.0, max_radius);
        // //    auto U[ip]=particles.vd()[ip];
        // //}

        // using namespace xt::placeholders;
        // auto U = xt::view(particles.vd(), xt::range(8, _));

        // std::cout << "V size: " << xt::adapt(V.shape()) << std::endl;
        // //scopi::type::velocity_t<dim> U=particles.vd();//??

        // std::cout<<"U size "<< xt::adapt(U.shape()) <<std::endl;

        // xt::xtensor<double, 1> Uc = xt::zeros<double>({U.size()*2});
        // std::size_t ii = 0;

        // for (auto& u: U)
        // {
        //     xt::view(Uc, xt::range(ii, ii + 2)) = u;
        //     ii += 2;
        // }
        // std::cout << "Uc: " << Uc << std::endl;
        // std::cout << "V^T@Uc: " << xt::linalg::dot(VT, Uc) << std::endl;
        // //std::cout << "Type of U: " << typeid(U).name() << std::endl;

        // xt::xtensor<double, 1> D = xt::zeros<double>({contacts2rom.size()});
        // std::size_t ic = 0;
        // for (auto& c: contacts2rom)
        // {
        //     D[ic++] = c.dij;
        // }

        // //size_t cols = W[0].size();
        // //std::cout<<"cols"<<cols<<end;
        // std::cout << "W^T@D: " << xt::linalg::dot(WT, D) << std::endl;
        // std::cout << "Qhat shape: " << xt::adapt(Qhat.shape()) << std::endl;
        // std::cout << "Bhat shape: " << xt::adapt(Bhat.shape()) << std::endl;
        // std::cout << "V^T@U shape: " << xt::adapt(xt::linalg::dot(VT, Uc).shape()) << std::endl;
        // std::cout << "W^T@D shape: " << xt::adapt(xt::linalg::dot(WT, D).shape()) << std::endl;
        // auto Chat=-(xt::linalg::dot(Bhat, xt::linalg::dot(VT, Uc)) - xt::linalg::dot(WT, D));
        // std::cout << "Chat: " << Chat << std::endl;

        // xt::xtensor<double, 1> lambda_hat = np.zeros<double>({W.shape(1)});// to compute with a pgd algorithm

        // auto new_velocities = xt::linalg::dot(V, xt::linalg::dot(V, Uc) - xt::linalg::dot(xt::transpose(Bhat), lambda_hat));
        // auto new_lambda = xt::linalg::dot(W, lambda_hat);

        // json_obj["contacts"] = {};

        // index = 0;
        // for (const auto& c : contacts2rom)
        // {
        //     json_obj["contacts"].push_back(c.to_json());
        //     json_obj["contacts"].back()["number"] = index++;
        // }

        // // Ajouter des données à l'objet JSON
        json_obj["lexit"].push_back(lexit); // add an element to the end of a container such that vector or list
        // add another field for the spontaneous velocity modulus
        json_obj["s"].push_back(s_vel);

        // Sauvegarder l'objet JSON dans un fichier
        if (flag_test == "train")
        {
            std::ofstream file("lexit_train.json"); // ofstream to open the file for writing
            if (file.is_open())
            {
                file << json_obj.dump(4); // The updated JSON object is serialized and written back to the file using json_obj.dump(4) (with
                                          // an indentation of 4 spaces for readability).
                file.close();
                std::cout << "Fichier JSON créé avec succès !" << std::endl;
            }
            else
            {
                std::cerr << "Erreur lors de l'ouverture du fichier !" << std::endl;
            }
        }
        else if (flag_test == "valid")
        {
            std::ofstream file("lexit_valid.json"); // ofstream to open the file for writing
            if (file.is_open())
            {
                file << json_obj.dump(4); // The updated JSON object is serialized and written back to the file using json_obj.dump(4) (with
                                          // an indentation of 4 spaces for readability).
                file.close();
                std::cout << "Fichier JSON créé avec succès !" << std::endl;
            }
            else
            {
                std::cerr << "Erreur lors de l'ouverture du fichier !" << std::endl;
            }
        }
    }

    return 0;
}

// Function definition
double computeDistance(double x1, double y1, double x2, double y2)
{
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}
