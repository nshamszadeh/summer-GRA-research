#ifndef FAULT_H
#define FAULT_H

#include <stdlib.h>
#include "reservoir.h"

// Various ways neurons/edges in the reservoir can fail.
enum fault_type {
    NONE,
    // TRANSIENT, // Brief change in randomly chosen neuron structure/behavior (potential application: simulating radioactive environment)
    // METASTABLE, // Fast, (possibly) random spiking of randomly chosen neuron.
    // EDGE_FAULT, // Random neuron's individual edge weight(s) set to 0
    // GLOBAL, // Gaussian pertubation vector multiplied to all nonzero synapses
    // CLAMP_HIGH, // Clamps a random neuron's weights and activation to V_th
    CLAMP_LOW // Clamps a random neuron's weights to 0 and activation to V_rest (or 0, not sure yet...)
};

struct neuron_state {
    double V;
    double spike;
};

/* Internal fault structure. 
   fault_type: Specifies what kind of fault this system simulates 
   failed_neuron: failed_neurons[i] == 1 => reservoir->neurons[i] has failed
   saved_neuron_states: Points to neurons pre-fault state (V, spike), used to restore functional neurons.
 */
 struct fault_system {
    enum fault_type fault_type;
    //double fault_probability; // applied to individual neurons in reservoir
    //size_t max_fault_count; // upper bound on number of failing neurons
    int *failed_neurons; // 
    struct neuron_state *saved_neuron_states; // Before initiating a fault on a neuron, save neuron state here to restore later
    double *W_restore;
};

struct fault_system *create_fault_system(struct reservoir *r, enum fault_type fault_type);
void free_fault_system(struct fault_system *fs);
void process_clamp_low(struct reservoir *r, struct fault_system *fs);
size_t get_victim_idx(struct reservoir *r, struct fault_system *fs);
size_t get_nearby_idx(size_t num_neurons, struct fault_system *fs, size_t neuron_idx);

#endif // FAULT_H