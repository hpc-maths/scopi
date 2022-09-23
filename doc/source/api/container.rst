scopi_container class
=====================

.. doxygenclass:: scopi::scopi_container
   :project: scopi
   :members:
   :protected-members:
<<<<<<< HEAD

Particles and objects
---------------------
::

    scopi_container<dim> particles;


Loop on positions of particles::
    
    for (std::size_t p = 0; p < particles.nb_active(); ++p)
    {
       particles.pos()(p) = /* ... */;
    }

Loop on objects::
    
    for (std::size_t o = 0; o < particles.size(); ++o)
    {
       particles[o] = /* ... */;
    }

Loop on particles, then access the object::

    for (std::size_t p = 0; p < particles.nb_active(); ++p)
    {
        particles[particles.object_index(p)] += /* ... */;
    }

Loop on positions of particles in object::

    for (std::size_t o = 0; o < particles.size(); ++o)
    {
        for (std::size_t p = 0; p < particles[0].size(); ++p)
        {
            particles.pos()(particles.offset(o) + p) = /* ... */;
        }
    }


.. graphviz::

    digraph G {
        node [shape=record];
        container [label="{<particles0>Particles|<objects0>Objects|Particles in an object} | {{0}|{0}|{0}} | {{1}|{}|{1}} | {{2}|{}|{2}} | {{3}|{}|{3}} | {{4}|{1}|{0}} | {{5}|{}|{1}} | {{6}|{}|{2}} | {{7}|{}|{3}} | {{<particles1>8}|{<objects1>}|{4}}"];
        container:objects0 -> container:particles0 [label="offset "]
        container:particles1 -> container:objects1 [label=" object_index"]
    }

