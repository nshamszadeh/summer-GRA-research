/**
 * @author Navid Shamszadeh
 * @author Tucker Mastin (Original mackey_glass_prediction.c provided with SpiresRC)
 * @brief Reservoir computing Mackey-Glass approximation with simulated fault injection.
 * Full ridge regression trains the output layer prior to inference, then online learning
 * retrains the output layer at each inference time step.
 * Use gnuplot to view a graph of predicted vs. target data
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <spires.h>

void generate_mackey_glass(double *buffer, size_t length, double x0, double tau, double beta, double gamma, int n)
{
    double mg_dt = 1.0;
    // Calculate the required length of the history buffer in discrete steps
    int history_len = (int)ceil(tau / mg_dt);

    // Allocate and initialize the history buffer
    double *history = malloc(history_len * sizeof(double));
    if (history == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for Mackey-Glass history.\n");
        return;
    }
    for (int i = 0; i < history_len; i++) {
        history[i] = x0;
    }

    buffer[0] = x0;
    double x_t = x0; // Current value of x

    for (size_t i = 0; i < length - 1; i++) {
        // Get the delayed value x(t - tau) from the history buffer
        double x_tau = history[i % history_len];

        // k1: slope at the beginning of the interval
        double k1 = mg_dt * (beta * x_tau / (1.0 + pow(x_tau, n)) - gamma * x_t);

        // k2: slope at the midpoint, using k1
        x_tau = history[(i + history_len / 2) % history_len]; // Approx. delay for midpoint
        double k2 = mg_dt * (beta * x_tau / (1.0 + pow(x_tau, n)) - gamma * (x_t + 0.5 * k1));

        // k3: slope at the midpoint, using k2
        double k3 = mg_dt * (beta * x_tau / (1.0 + pow(x_tau, n)) - gamma * (x_t + 0.5 * k2));

        // k4: slope at the end of the interval, using k3
        x_tau = history[(i + 1) % history_len]; // Approx. delay for endpoint
        double k4 = mg_dt * (beta * x_tau / (1.0 + pow(x_tau, n)) - gamma * (x_t + k3));

        // Update the current value using the weighted average of the slopes
        x_t += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

        // Store the new value in the history buffer (for future delayed lookups)
        history[(i + 1) % history_len] = x_t;

        // Store the new value in the output buffer
        buffer[i + 1] = x_t;
    }

    free(history);
}

int random_fault(spires_reservoir *r, spires_fault_system *fs, double fault_probability)
{
    double probability = (double)rand()/RAND_MAX;
    if (probability <= fault_probability) {
        spires_generate_fault(r, fs);
        return 1;
    }
    return 0;
}

/* Holds the error metrics produced by compute_fault_error_metrics(). */
typedef struct {
    double rmse_overall;       /* RMSE across the entire series */
    double rmse_at_fault;      /* RMSE restricted to timesteps where a fault fired */
    double rmse_not_at_fault;  /* RMSE restricted to timesteps where no fault fired */
    double rmse_post_fault;    /* RMSE within `window` steps after a fault (inclusive of the fault step) */
    double rmse_unaffected;    /* RMSE outside that post-fault window */
    size_t n_overall;
    size_t n_at_fault;
    size_t n_not_at_fault;
    size_t n_post_fault;
    size_t n_unaffected;
} fault_error_metrics;

/*
 * Computes RMSE-based error metrics comparing reservoir_outputs[i] against
 * the true Mackey-Glass target input_series[i + horizon], split three ways:
 *
 *   1. Overall RMSE across the whole series.
 *   2. RMSE keyed on fault_times[i] (instant impact: did a fault fire on
 *      exactly this step?).
 *   3. Windowed post-fault RMSE: RMSE over the `window` steps following
 *      (and including) each fault, vs. RMSE over all other steps. This
 *      captures lingering degradation that a single-step comparison misses.
 *
 * online_training_enabled does not change the arithmetic — it only labels the
 * printed output — so callers can correctly tag results when comparing the
 * online-on vs. online-off conditions.
 *
 * series_length, timesteps, and horizon mirror the values used elsewhere in
 * main(): target_idx = i + horizon must stay below timesteps for a sample
 * to be valid (this excludes the last `horizon` steps, which have no true
 * target available).
 */
fault_error_metrics compute_fault_error_metrics(
    const double *reservoir_outputs,
    const double *input_series,
    const size_t *fault_times,
    size_t series_length,
    size_t timesteps,
    size_t horizon,
    size_t window,
    int online_training_enabled)
{
    fault_error_metrics m = {0};

    double sum_sq_overall = 0.0;
    double sum_sq_at_fault = 0.0, sum_sq_not_at_fault = 0.0;
    double sum_sq_post_fault = 0.0, sum_sq_unaffected = 0.0;

    /* Mark every step within `window` steps after (and including) a fault. */
    int *affected = calloc(series_length, sizeof(int));
    if (affected == NULL) {
        fprintf(stderr, "compute_fault_error_metrics: failed to allocate 'affected' array\n");
        return m; /* zeroed-out metrics signal failure */
    }
    for (size_t i = 0; i < series_length; i++) {
        if (fault_times[i]) {
            size_t end = (i + window < series_length) ? i + window : series_length;
            for (size_t j = i; j < end; j++) affected[j] = 1;
        }
    }

    for (size_t i = 0; i < series_length; i++) {
        size_t target_idx = i + horizon;
        if (target_idx >= timesteps) continue; /* no valid target for this step */

        double err = reservoir_outputs[i] - input_series[target_idx];
        double sq_err = err * err;

        sum_sq_overall += sq_err;
        m.n_overall++;

        if (fault_times[i]) {
            sum_sq_at_fault += sq_err;
            m.n_at_fault++;
        } else {
            sum_sq_not_at_fault += sq_err;
            m.n_not_at_fault++;
        }

        if (affected[i]) {
            sum_sq_post_fault += sq_err;
            m.n_post_fault++;
        } else {
            sum_sq_unaffected += sq_err;
            m.n_unaffected++;
        }
    }

    free(affected);

    m.rmse_overall      = m.n_overall      ? sqrt(sum_sq_overall      / m.n_overall)      : NAN;
    m.rmse_at_fault     = m.n_at_fault     ? sqrt(sum_sq_at_fault     / m.n_at_fault)     : NAN;
    m.rmse_not_at_fault = m.n_not_at_fault ? sqrt(sum_sq_not_at_fault / m.n_not_at_fault) : NAN;
    m.rmse_post_fault   = m.n_post_fault   ? sqrt(sum_sq_post_fault   / m.n_post_fault)   : NAN;
    m.rmse_unaffected   = m.n_unaffected   ? sqrt(sum_sq_unaffected   / m.n_unaffected)   : NAN;

    printf("\n--- Prediction Error (online_learning = %s, window = %zu) ---\n",
           online_training_enabled ? "ON" : "OFF", window);
    printf("Overall RMSE:                  %f (n=%zu)\n", m.rmse_overall, m.n_overall);
    printf("RMSE at fault timesteps:       %f (n=%zu)\n", m.rmse_at_fault, m.n_at_fault);
    printf("RMSE at non-fault timesteps:   %f (n=%zu)\n", m.rmse_not_at_fault, m.n_not_at_fault);
    printf("RMSE within %zu steps of fault: %f (n=%zu)\n", window, m.rmse_post_fault, m.n_post_fault);
    printf("RMSE unaffected steps:         %f (n=%zu)\n", m.rmse_unaffected, m.n_unaffected);

    return m;
}

int main(void) 
{
    srand(time(NULL));
    FILE *output_file = fopen("data/output_signals.dat", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error: Could not open data files for writing.\n");
        return 1;
    }

    /* ---- Generate Mackey–Glass ---- */
    size_t timesteps = 1000;
    double x0 = 0.2;
    double tau = 22;       // Must be > 17 for chaos
    double beta = 0.4;
    double gamma = 0.2;
    int n = 11;

    double *input_series = malloc(timesteps * sizeof(double));
    if (input_series == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for input_series.\n");
        fclose(output_file);
        return 1;
    }
    
    generate_mackey_glass(input_series, timesteps, x0, tau, beta, gamma, n);


    //int rc;
    const size_t Din  = 1;  /* Mackey–Glass is scalar input */
    const size_t Dout = 1;  /* predict next value */
    const size_t horizon = 2;

    const size_t series_length  = timesteps - horizon;
    
    /* Build target series: y[t] = x[t + horizon].
     * We optimize/train on T = series_length - horizon samples. */
    if (series_length <= horizon) {
            fprintf(stderr, "series too short for horizon=%zu\n", horizon);
            return 1;
    }
    size_t T = series_length;

    double *target_series = malloc(T * Dout * sizeof(double));
    if (!target_series) {
            fprintf(stderr, "alloc failure for target_series\n");
            return 1;
    }
    for (size_t t = 0; t < T; t++)
            target_series[t] = input_series[t + horizon];

    /* ---- Neuron Hyperparameters ---- */
    double neuron_params[] = {
        0.0,    // V_0
        1.0,    // V_th
        0.5,    // leak_rate
        0.1    // bias
    };
    
    /* Base config (“ball-park” defaults) */
    spires_reservoir_config base = {
            .num_neurons       = 1000,
            .num_inputs        = Din,
            .num_outputs       = Dout,
            .spectral_radius   = 0.9,
            .ei_ratio          = 0.50,
            .input_strength    = 1.00,
            .connectivity      = 0.01,
            .dt                = 0.1,
            .connectivity_type = SPIRES_CONN_RANDOM,
            .neuron_type       = SPIRES_NEURON_LIF_DISCRETE,
            .neuron_params     = neuron_params,   /* alpha will be set internally by optimizer */
    };

    /* Train final reservoir with the best config and ridge on ALL T samples */
    spires_reservoir *R = NULL;
    if (spires_reservoir_create(&base, &R) != SPIRES_OK || !R) {
            fprintf(stderr, "failed to create reservoir\n");
            free(target_series);
            return 1;
    }

    const double lambda = pow(10.0, 0.1);
    if (spires_train_ridge(R, input_series, target_series, T, lambda) != SPIRES_OK) {
            fprintf(stderr, "ridge training failed\n");
            spires_reservoir_destroy(R);
            free(target_series);
            free(input_series);
            fclose(output_file);
            return 1;
    }

    spires_reservoir_reset(R);
    spires_fault_system *fs = NULL;
    if (spires_create_fault_system(R, SPIRES_FAULT_CLAMP_LOW, &fs)) {
        fprintf(stderr, "failed to create fault system\n");
        free(target_series);
        free(input_series);
        spires_reservoir_destroy(R);
        fclose(output_file);
        return 1;
    }

    size_t *fault_times = calloc(series_length, sizeof(size_t));
    if (fault_times == NULL) {
        fprintf(stderr, "failed to create fault_times array\n");
        free(target_series);
        free(input_series);
        spires_reservoir_destroy(R);
        fclose(output_file);
        return 1;
    }

    size_t fault_count = 0;
    double learning_rate = 1e-3;
    int online_learning_enabled = 0;
    double fault_probability = 0.05;

    double *reservoir_outputs = malloc(Dout * series_length * sizeof(double));
    if (!reservoir_outputs) {
        fprintf(stderr, "failed to allocate reservoir_outputs\n");
        spires_reservoir_destroy(R);
        spires_destroy_fault_system(fs);
        free(input_series);
        free(target_series);
        fclose(output_file);
        return 1;
    }

    for (size_t i = 0; i < Dout * series_length; i++) {
        const double *current_input = &input_series[i * Din];

        if (i > Dout * series_length / 2) { // only inject faults halfway through dataset
            if (random_fault(R, fs, fault_probability)) {
                printf("Fault injected at timestep t = %lu\n", i);
                // save timestep of fault
                fault_times[i] = 1;
                fault_count += 1;
            }
        }

        spires_step(R, current_input);
            
        if (online_learning_enabled) {
            if (spires_train_online(R, &target_series[i * Dout], learning_rate) != SPIRES_OK) {
                fprintf(stderr, "online training failed at t = %zu\n", i);
                return 1;
            }
        }
        spires_compute_output(R, &reservoir_outputs[i * Dout]);
    }

    
    printf("Fault count: %lu\n", fault_count);
    compute_fault_error_metrics(reservoir_outputs, input_series, fault_times, series_length, timesteps,
                                horizon, 10, online_learning_enabled);


    fprintf(output_file, "# Timestep Input_Signal Reservoir_Output Fault\n");
    for (size_t i = 0; i < series_length; i++) {
        fprintf(output_file, "%zu %f %f %zu\n", i, input_series[i], reservoir_outputs[i], fault_times[i]);
    }

    free(reservoir_outputs);
    free(target_series);
    spires_destroy_fault_system(fs);
    spires_reservoir_destroy(R);
    fclose(output_file);
    free(input_series);
    return 0;
}

