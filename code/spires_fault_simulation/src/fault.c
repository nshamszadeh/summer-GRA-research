#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "fault.h"

struct fault_system* create_fault_system(struct reservoir *r, enum fault_type fault_type)
{
    struct fault_system *fs = malloc(sizeof(struct fault_system));
    if (fs == NULL) {
        fprintf(stderr, "Error allocating memory for fault_system.\n");
        return NULL;
    }
    // set the fault type
    fs->fault_type = fault_type;

    // save weight matrix pre-fault:
	size_t W_size = r->num_neurons * r->num_neurons;
	fs->W_restore = malloc(sizeof(double) * W_size);
	if ((fs->W_restore) == NULL) {
		fprintf(stderr, "Error allocating memory for W_restore.\n");
        free(fs);
		return NULL;
	}
	memcpy(fs->W_restore, r->W, W_size);

	// allocate memory for saved neuron information
	fs->failed_neurons = calloc(r->num_neurons, sizeof(int)); // calloc to zero-initialize the array
	if ((fs->failed_neurons) == NULL) {
		fprintf(stderr, "Error allocating memory for failed_neurons.\n");
        free(fs->W_restore);
        free(fs);
		return NULL;
	}
	fs->saved_neuron_states = calloc(r->num_neurons, sizeof(struct neuron_state));
	if ((fs->saved_neuron_states) == NULL) {
		fprintf(stderr, "Error allocating memory for saved_neuron_states.\n");
		free(fs->W_restore);
        free(fs->failed_neurons);
        free(fs);
        return NULL;
	}
    return fs;
}

void free_fault_system(struct fault_system *fs)
{
    if (!fs)
        return;
    free(fs->W_restore);
    free(fs->saved_neuron_states);
    free(fs->failed_neurons);
    fs->W_restore = NULL;
    fs->saved_neuron_states = NULL;
    fs->failed_neurons = NULL;
    free(fs);
    fs = NULL;
}

// returns index of working neuron near neuron_idx in reservoir
// if there are no working neurons, returns num_neurons+1
size_t get_nearby_idx(size_t num_neurons, struct fault_system *fs, size_t neuron_idx)
{
    // neuron is already killed, find a nearby functioning neuron
    // search for nearby functioning neuron by indexing up and down
    size_t low_idx = SIZE_MAX;  // SIZE_MAX means "not found"
    for (size_t i = neuron_idx; i-- > 0; ) {
        if (!fs->failed_neurons[i]) {
            low_idx = i;
            break;
        }
    }

    size_t high_idx = SIZE_MAX; // SIZE_MAX means "not found"
    for (size_t i = neuron_idx + 1; i < num_neurons; i++) {
        if (!fs->failed_neurons[i]) {
            high_idx = i;
            break;
        }
    }

    if (low_idx == SIZE_MAX && high_idx == SIZE_MAX) return num_neurons + 1;
    if (low_idx == SIZE_MAX) return high_idx;
    if (high_idx == SIZE_MAX) return low_idx;

    // both found — return whichever is closer to neuron_idx
    size_t low_dist = neuron_idx - low_idx;
    size_t high_dist = high_idx - neuron_idx;
    return (low_dist <= high_dist) ? low_idx : high_idx;
}

// Returns index of random non-faulty neuron
size_t get_victim_idx(struct reservoir *r, struct fault_system *fs)
{
    size_t neuron_idx = (size_t)rand() / (RAND_MAX / r->num_neurons + 1);
    if (fs->failed_neurons[neuron_idx]) {
        neuron_idx = get_nearby_idx(r->num_neurons, fs, neuron_idx);
    }
    return neuron_idx;
}

/* Picks a random neuron, sets its edge weights to 0 and its state to V_reset/V_0 */
void process_clamp_low(struct reservoir *r, struct fault_system *fs)
{
    // pick a random neuron in the reservoir
    size_t neuron_idx = get_victim_idx(r, fs);
    if (neuron_idx == r->num_neurons + 1) return; // no available neurons to kill
    // clamping logic is dependent on neuron type
    switch(r->neuron_type) {
        case LIF_DISCRETE: {
            struct lif_discrete_neuron *neuron = (struct lif_discrete_neuron *)(r->neurons[neuron_idx]);
            // save neuron state
            fs->failed_neurons[neuron_idx] = 1;
            fs->saved_neuron_states[neuron_idx] = (struct neuron_state){neuron->V, neuron->spike};
            neuron->V = neuron->V_0;
            break;
        }
        case LIF_BIO: {
            struct lif_bio_neuron *neuron = (struct lif_bio_neuron *)(r->neurons[neuron_idx]);
            neuron->V = neuron->V_0;
            break;
        }
        case FLIF_CAPUTO: {
            struct flif_caputo_neuron *neuron = (struct flif_caputo_neuron *)(r->neurons[neuron_idx]);
            neuron->V = neuron->V_reset;
            break;
        }
        case FLIF_GL: {
            struct flif_gl_neuron *neuron = (struct flif_gl_neuron *)(r->neurons[neuron_idx]);
            neuron->V = neuron->V_reset;
            break;
        }
        case FLIF_DIFFUSIVE: {
            struct flif_diffusive_neuron *neuron = (struct flif_diffusive_neuron *)(r->neurons[neuron_idx]);
            neuron->V = neuron->V_reset;
            break;
        }
        default: {
            fprintf(stderr, "Error, unknown neuron type: %d.\n", (int)r->neuron_type); // technically redundant given spires_generate_fault()
            return;
        }
    }
    // zero all incoming and outgoing edge weights
    for (size_t i = 0; i < r->num_neurons; i++) {
        r->W[r->num_neurons * neuron_idx + i] = 0.0;
        r->W[r->num_neurons * i + neuron_idx] = 0.0;
    }
}
