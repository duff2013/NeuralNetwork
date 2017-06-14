//
//  NeuralNetwork.hpp
//  Teensy
//
//  Created by Colin on 4/10/17.
//  Copyright © 2017 Colin Duffy. All rights reserved.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include "Arduino.h"
#include "FixedPoint.hpp"

template<const uint16_t IN, const uint16_t HN, const uint16_t ON>
class NeuralNetwork {
private:
    float Hidden[HN];
    float Output[ON];
    float HiddenWeights[IN + 1][HN];
    float output_weights[HN + 1][ON];
    float hidden_delta[HN];
    float output_delta[ON];
    float change_hidden_weights[IN + 1][HN];
    float change_output_weights[HN + 1][ON];
    float LearningRate;// = .00000001;
    const float Momentum = 0.00001;
    const float InitialWeightMax = 0.0005;
    const float Success = .0004;
    float Error;
public:
    NeuralNetwork( void ) { }
    
    NeuralNetwork( uint8_t channel ) {
        randomSeed( analogRead( channel ) );
        for ( int i = 0 ; i < HN ; i++ ) {
            for ( int j = 0 ; j <= IN ; j++ ) {
                float rnd = float( random( 100 ) ) / 100;
                change_hidden_weights[j][i] = 0.0 ;
                HiddenWeights[j][i] = 2.0 * ( rnd - 0.5 ) * InitialWeightMax ;
            }
        }

        randomSeed( analogRead( channel ) );
        for ( int i = 0 ; i < ON ; i ++ ) {
            for ( int j = 0 ; j <= HN ; j++ ) {
                float rnd = float( random( 100 ) ) / 100;
                change_output_weights[j][i] = 0.0 ;
                output_weights[j][i] = 2.0 * ( rnd - 0.5 ) * InitialWeightMax ;
            }
        }
        LearningRate     = 0.001;
        //Momentum         = 0.09;
        //InitialWeightMax = 0.5;
        //Success          = 0.0004;
    }
    
    // ----------------------------------------------------------------------------------------
    void setLearningRate( float rate ) {
        LearningRate = rate;
    }
    
    // ----------------------------------------------------------------------------------------
    void setMomentum( float momentum ) {
        //Momentum = momentum;
    }
    
    // ----------------------------------------------------------------------------------------
    void setSuccessError( float success_error ) {
        //Success = success_error;
    }
    
    // ----------------------------------------------------------------------------------------
    void setInitialWeightMax( float weight_max ) {
        //InitialWeightMax = weight_max;
    }
    
    // ----------------------------------------------------------------------------------------
    virtual float acivationFunction( float data ) {
        return  1.0 / ( 1.0 + exp( -data ) );
    }
    // ----------------------------------------------------------------------------------------
    void computeNetworkOutput( const float *input_data, byte *output_data ) {
        /******************************************************************
         Compute hidden layer activations
         ******************************************************************/
        float Accum = 0.0;
        for ( int i = 0 ; i < HN; i++ ) {
            Accum = HiddenWeights[IN][i] ;
            for ( int j = 0 ; j < IN ; j++ ) {
                Accum += input_data[j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0 / ( 1.0 + exp( -Accum ) );
        }
        /******************************************************************
         Compute output layer activations and calculate errors
         ******************************************************************/
        for ( int i = 0 ; i < ON ; i++ ) {
            Accum = output_weights[HN][i] ;
            for ( int j = 0 ; j < HN; j++ ) {
                Accum += Hidden[j] * output_weights[j][i] ;
            }
            Output[i] = 1.0 / ( 1.0 + exp( -Accum ) );
            output_data[i] = roundf(Output[i]);
        }
    }
    // ----------------------------------------------------------------------------------------
    float computeNetworkOutput( const float input_data[][IN], const float target_data[][ON], float output_data[][ON], uint8_t pattern_count ) {
        float error = 0.0;
        for ( int p = 0; p < pattern_count; p++ ) {
            /******************************************************************
             Compute hidden layer activations
             ******************************************************************/
            float Accum = 0.0;
            for ( int i = 0 ; i < HN; i++ ) {
                Accum = HiddenWeights[IN][i] ;
                for ( int j = 0 ; j < IN ; j++ ) {
                    Accum += input_data[p][j] * HiddenWeights[j][i] ;
                }
                Hidden[i] = 1.0 / ( 1.0 + exp( -Accum ) );
            }
            /******************************************************************
             Compute output layer activations and calculate errors
             ******************************************************************/
            for ( int i = 0 ; i < ON ; i++ ) {
                Accum = output_weights[HN][i] ;
                for ( int j = 0 ; j < HN; j++ ) {
                    Accum += Hidden[j] * output_weights[j][i] ;
                }
                Output[i] = 1.0 / ( 1.0 + exp( -Accum ) );
                output_data[p][i] = Output[i];
                error += 0.5 * ( target_data[p][i] - Output[i] ) * ( target_data[p][i] - Output[i] );
            }
        }
        return error;
    }
    
    // ----------------------------------------------------------------------------------------
    float trainNetwork( const float input_data[][IN], const float target_data[][ON], uint8_t pattern_count ) {
        /******************************************************************
         Randomize order of training patterns
         ******************************************************************/
        uint16_t RandomizedIndex[pattern_count];
        
        for ( int p = 0 ; p < pattern_count ; p++ ) {
            RandomizedIndex[p] = p ;
        }
        
        for ( int p = 0 ; p < pattern_count; p++) {
            int q = random( pattern_count );
            int r = RandomizedIndex[p] ;
            RandomizedIndex[p] = RandomizedIndex[q] ;
            RandomizedIndex[q] = r ;
        }
        
        float error = 0.0;
        
        for ( int q = 0; q < pattern_count; q++ ) {
            
            int p = RandomizedIndex[q];
            float Accum = 0.0;
            /******************************************************************
             Compute hidden layer activations
             ******************************************************************/
            for ( int i = 0; i < HN; i++ ) {
                Accum = HiddenWeights[IN][i] ;
                //Serial.printf("Accum1: %f\n", Accum);
                for ( int j = 0; j < IN; j++ ) {
                    Accum += input_data[p][j] * HiddenWeights[j][i];
                    //Serial.printf("input_data[%02i][%02i]: %i\n", p, j, input_data[p][j]);
                }
                Hidden[i] = 1.0 / ( 1.0 + exp( -Accum ) );
                //Serial.printf("Hidden[%02i]: %f | Accum: %f\n", i, Hidden[i], Accum);
            }
            
            
            /******************************************************************
             Compute output layer activations and calculate errors
             ******************************************************************/
            for ( int i = 0; i < ON; i++ ) {
                Accum = output_weights[HN][i];
                //Serial.printf("Accum2: %f\n", Accum);
                for ( int j = 0; j < HN; j++ ) {
                    Accum += Hidden[j] * output_weights[j][i];
                    //Serial.printf("Accum2: %f\n", Accum);
                }
                Output[i] = 1.0 / ( 1.0 + exp( -Accum ) );
                //Serial.printf("Output[%02i]: %f\n", i, Output[i]);
                output_delta[i] = ( target_data[p][i] - Output[i] ) * Output[i] * ( 1.0 - Output[i] );
                //output_delta[i] = ( target_data[p][i] - Output[i] );
                error += 0.5 * ( target_data[p][i] - Output[i] ) * ( target_data[p][i] - Output[i] );
                //error -= ( target_data[p][i] * log( Output[i] ) + ( 1.0 - target_data[p][i] ) * log( 1.0 - Output[i] ) ) ;
                //Serial.printf("output_delta[%02i]: %f | Output[%02i]: %f | error: %f\n", i, output_delta[i], i, Output[i], Error);
            }
            //Serial.println();
            /******************************************************************
             Backpropagate errors to hidden layer
             ******************************************************************/
            for ( int i = 0; i < HN; i++ ) {
                Accum = 0.0;
                for ( int j = 0; j < ON; j++ ) {
                    Accum += output_weights[i][j] * output_delta[j];
                }
                hidden_delta[i] = Accum * Hidden[i] * ( 1.0 - Hidden[i] );
                //Serial.printf("hidden_delta[%02i]: %f | Accum: %f\n", i, hidden_delta[i], Accum);
            }
            /******************************************************************
             Update Inner-->Hidden Weights
             ******************************************************************/
            for ( int i = 0; i < HN; i++ ) {
                change_hidden_weights[IN][i] = LearningRate * hidden_delta[i] + Momentum * change_hidden_weights[IN][i];
                HiddenWeights[IN][i] += change_hidden_weights[IN][i];
                for ( int j = 0; j < IN; j++ ) {
                    change_hidden_weights[j][i] = LearningRate * input_data[p][j] * hidden_delta[i] + Momentum * change_hidden_weights[j][i];
                    HiddenWeights[j][i] += change_hidden_weights[j][i];
                }
            }
            
            /******************************************************************
             Update Hidden-->Output Weights
             ******************************************************************/
            for ( int i = 0; i < ON; i ++ ) {
                change_output_weights[HN][i] = LearningRate * output_delta[i] + Momentum * change_output_weights[HN][i];
                output_weights[HN][i] += change_output_weights[HN][i];
                for ( int j = 0 ; j < HN ; j++ ) {
                    change_output_weights[j][i] = LearningRate * Hidden[j] * output_delta[i] + Momentum * change_output_weights[j][i];
                    output_weights[j][i] += change_output_weights[j][i];
                }
            }
            
            
        }
        //Serial.println("-------------------------------------------");
        return error;
    }
    // ----------------------------------------------------------------------------------------
    float trainNetwork( const float *input_data, const float *target_data ) {
        float error = 0.0;
        
        float Accum = 0.0;
        /******************************************************************
         Compute hidden layer activations
         ******************************************************************/
        for ( int i = 0; i < HN; i++ ) {
            Accum = HiddenWeights[IN][i] ;
            for ( int j = 0; j < IN; j++ ) {
                Accum += input_data[j] * HiddenWeights[j][i];
            }
            Hidden[i] = 1.0 / ( 1.0 + exp( -Accum ) );
        }
        /******************************************************************
         Compute output layer activations and calculate errors
         ******************************************************************/
        for ( int i = 0; i < ON; i++ ) {
            Accum = output_weights[HN][i];
            for ( int j = 0; j < HN; j++ ) {
                Accum += Hidden[j] * output_weights[j][i];
            }
            Output[i] = 1.0 / ( 1.0 + exp( -Accum ) );
            output_delta[i] = ( target_data[i] - Output[i] ) * Output[i] * ( 1.0 - Output[i] );
            //output_delta[i] = ( target_data[i] - Output[i] );
            error += 0.5 * ( target_data[i] - Output[i] ) * ( target_data[i] - Output[i] );
            //error -= ( target_data[i] * log( Output[i] ) + ( 1.0 - target_data[i] ) * log( 1.0 - Output[i] ) ) ;
        }
        /******************************************************************
         Backpropagate errors to hidden layer
         ******************************************************************/
        for ( int i = 0; i < HN; i++ ) {
            Accum = 0.0;
            for ( int j = 0; j < ON; j++ ) {
                Accum += output_weights[i][j] * output_delta[j];
            }
            hidden_delta[i] = Accum * Hidden[i] * ( 1.0 - Hidden[i] );
        }
        /******************************************************************
         Update Inner-->Hidden Weights
         ******************************************************************/
        for ( int i = 0; i < HN; i++ ) {
            change_hidden_weights[IN][i] = LearningRate * hidden_delta[i] + Momentum * change_hidden_weights[IN][i];
            HiddenWeights[IN][i] += change_hidden_weights[IN][i];
            for ( int j = 0; j < IN; j++ ) {
                change_hidden_weights[j][i] = LearningRate * input_data[j] * hidden_delta[i] + Momentum * change_hidden_weights[j][i];
                HiddenWeights[j][i] += change_hidden_weights[j][i];
            }
        }
        /******************************************************************
         Update Hidden-->Output Weights
         ******************************************************************/
        for ( int i = 0; i < ON; i ++ ) {
            change_output_weights[HN][i] = LearningRate * output_delta[i] + Momentum * change_output_weights[HN][i];
            output_weights[HN][i] += change_output_weights[HN][i];
            for ( int j = 0 ; j < HN ; j++ ) {
                change_output_weights[j][i] = LearningRate * Hidden[j] * output_delta[i] + Momentum * change_output_weights[j][i];
                output_weights[j][i] += change_output_weights[j][i];
            }
        }
        return error;
    }
};


/*class Axon;
class Neuron;

//template<unsigned int num_of_axons>
class Axon {
private:
    float weight;
    Neuron *axon_input;
    Neuron *axon_output;
public:
    Axon ( void ) {
        axon_input = NULL;
        axon_output = NULL;
        randomSeed( analogRead( 3 ) );
        float rnd = float(random(100)) / 100;
        weight = .5;//2.0 * ( rnd - 0.5 ) * 0.5;
        prev_weight = 0.0;
        //Serial.printf("Axon Weight: %f\n", weight);
    }
    
    void set_input_neuron( Neuron *n ) {
        axon_input = n;
    }
    
    void set_output_neuron( Neuron *n ) {
        axon_output = n;
    }
    
    Neuron *get_input_neuron( void ) {
        return axon_input;
    }
    
    Neuron *get_output_neuron( void ) {
        return axon_output;
    }
    
    void set_weight( float w ) {
        weight = w;
    }
    
    float get_weight( void ) {
        return weight;
    }
    
    float prev_weight;
};

class Neuron {
private:
    uint16_t axon_count;
public:
    Neuron ( void ) :   input_data_value( 0.00 ),
                        output_data_value( 1.00 ),
                        target_data_value( 0.00 ),
                        prev_weight( 1.00 ),
                        axon_count( 0 )
    {
        //Serial.printf("Neuron: %p\n", this);
    }
    float delta;
    float prev_weight;
    float input_data_value;
    float output_data_value;
    float target_data_value;
    
    static float compute_activation( float accum_weights ) {
        return 1.0 / ( 1.0 + exp( -accum_weights ) );
    }
    
    void update_axon_count( uint16_t num ) {
        
    }
    
};

template<unsigned int in_num_nron, unsigned int hid_num_nron, unsigned int out_num_nron>
class NeuralNetwork {
private:
    Neuron Input_Neuron[in_num_nron + 1];
    Neuron Hidden_Neuron[hid_num_nron + 1];
    Neuron Output_Neuron[out_num_nron];
    Axon   In_Hid_Axon[( in_num_nron + 1 ) * ( hid_num_nron )];
    Axon   Hid_Out_Axon[( hid_num_nron + 1 ) * out_num_nron];
    int input_neuron_count;
    int hidden_neuron_count;
    int output_neuron_count;
    int in_hid_axon_count;
    int hid_out_axon_count;
    float learning_rate;
    float momentum;
    float error;
    float *target_data;
    float *output_data;
public:
    
    NeuralNetwork( void ) : input_neuron_count( in_num_nron + 1 ),
                            hidden_neuron_count( hid_num_nron + 1 ),
                            output_neuron_count( out_num_nron ),
                            in_hid_axon_count( ( in_num_nron + 1 ) * ( hid_num_nron ) ),
                            hid_out_axon_count( ( hid_num_nron + 1 ) * out_num_nron )
    {
        int idx = 0;
        for (int i = 0; i < input_neuron_count; i++) {
            for ( int j = 0; j < hidden_neuron_count - 1; j++ ) {
                In_Hid_Axon[idx].set_input_neuron( &Input_Neuron[i] );
                In_Hid_Axon[idx++].set_output_neuron( &Hidden_Neuron[j] );
            }
        }
        
        // ---------------------------------------------------------------------------------
        
        idx = 0;
        for (int i = 0; i < hidden_neuron_count; i++) {
            for ( int j = 0; j < output_neuron_count; j++ ) {
                Hid_Out_Axon[idx].set_input_neuron( &Hidden_Neuron[i] );
                Hid_Out_Axon[idx++].set_output_neuron( &Output_Neuron[j] );
            }
        }
        
        // ---------------------------------------------------------------------------------
        
        idx = in_hid_axon_count - 1;
        for (int i = 0; i < input_neuron_count + 1; i++) {
            //Serial.println(idx);
            In_Hid_Axon[idx--].set_weight( 0.0 );
        }

        
        for (int i = 0; i < in_hid_axon_count; i++) {
            //Serial.printf("in_hid_Axon[%02i]  Address: %p | Input Neuron: %p | Hidden Neuron: %p\n",i, &In_Hid_Axon[i], In_Hid_Axon[i].get_input_neuron(), In_Hid_Axon[i].get_output_neuron());
        }
        
        Serial.println();
        for (int i = 0; i < hid_out_axon_count; i++) {
            //Serial.printf("hid_out_Axon[%02i] Address: %p | Hidden Neuron: %p | Output Neuron: %p\n",i, &Hid_Out_Axon[i], Hid_Out_Axon[i].get_input_neuron(), Hid_Out_Axon[i].get_output_neuron());
        }
        
        learning_rate = 0.1;
        momentum      = .9;
        error         = 0.0;
    }
    
    template <typename T>
    void input( T *input_array ) {
        for ( int i = 0; i < input_neuron_count - 1; i++ ) {
            Input_Neuron[i].input_data_value  = input_array[i];
            Input_Neuron[i].output_data_value = input_array[i];
        }
    }
    
    template <typename T>
    void target( T *target_array ) {
        //target_data = static_cast<float*>( target_array );
        for ( int i = 0; i < output_neuron_count; i++ ) {
            Output_Neuron[i].target_data_value = target_array[i];
        }
    }
    
    template <typename T>
    void output( T *output_array ) {
        output_data = static_cast<float*>( output_array );
    }
    
    void computeNetwork( void ) {
        //Serial.println();
        float Accum = 0.0;
        for (int i = in_hid_axon_count - 1; i >= 0; i--) {
            Neuron *in  = In_Hid_Axon[i].get_input_neuron( );
            Neuron *out = In_Hid_Axon[i].get_output_neuron( );
            
            float data   = in->output_data_value;
            float weight = In_Hid_Axon[i].get_weight( );
            
            float outval = out->input_data_value;
            
            out->input_data_value += data * weight;
            //Serial.printf("Input Neuron: %p | Hidden Neuron: %p | data: %f | weight: %f | out data: %f | outval: %f\n", in, out, data, weight, out->input_data_value, outval);
        }
        
        //Serial.println();
        //compute_activation( float accum_weights )
        for ( int i = 0; i < hidden_neuron_count - 1; i++ ) {
            float data = Hidden_Neuron[i].input_data_value;
            float d1 = data;
            data = Hidden_Neuron[i].compute_activation( data );
            Hidden_Neuron[i].output_data_value = data;
            //Serial.printf("data: %f | d1: %f\n", data, d1);
        }
        
        //Serial.println();
        for (int i = hid_out_axon_count - 1; i >= 0; i--) {
            Neuron *in  = Hid_Out_Axon[i].get_input_neuron( );
            Neuron *out = Hid_Out_Axon[i].get_output_neuron( );
            float data   = in->output_data_value;
            float weight = Hid_Out_Axon[i].get_weight( );
            out->input_data_value += data * weight;
            //Serial.printf("Hidden Neuron: %p | Output Neuron: %p | data: %f | weight: %f | out data: %f\n", in, out, data, weight, out->input_data_value);
        }

        //Serial.println();
        float error = 0.0;
        for ( int i = 0; i < output_neuron_count; i++ ) {
            float data = Output_Neuron[i].input_data_value;
            data = Output_Neuron[i].compute_activation( data );
            Output_Neuron[i].output_data_value = data;
            //Serial.printf("data: %f\n", data);
            float target = Output_Neuron[i].target_data_value;
            float delta = Output_Neuron[i].delta;
            delta = ( target - data ) * data * ( 1.0 - data );
            Output_Neuron[i].delta = delta;
            error += 0.5 * ( target - data ) * ( target - data );
            //Serial.printf("delta: %f | error: %f\n", delta, error);
        }
    }
    // -------------------------------------------------------------------------------------
    void trainNetwork( void ) {
        computeNetwork( );
        float accum = 0.0;
        for (int i = 0; i < hid_out_axon_count; i++) {
            Neuron *in   = Hid_Out_Axon[i].get_input_neuron( );
            Neuron *out  = Hid_Out_Axon[i].get_output_neuron( );
            float weight = Hid_Out_Axon[i].get_weight( );
            float delta  = out->delta;
            accum += weight * delta;
        }
        
        for ( int i = 0; i < hidden_neuron_count; i++ ) {
            float delta = Hidden_Neuron[i].delta;
            float data = Hidden_Neuron[i].output_data_value;
            delta = accum * data * ( 1.0 - data );
            Hidden_Neuron[i].delta = delta;
            //Serial.printf("delta: %f | accum: %f\n", delta, accum);
        }
        // ----------------------------------------------------------------
        //Serial.println(in_hid_axon_count);
        for (int i = 0; i < in_hid_axon_count; i++) {
            Neuron *in  = In_Hid_Axon[i].get_input_neuron( );
            Neuron *out = In_Hid_Axon[i].get_output_neuron( );
            
            float input  = in->output_data_value;
            float weight = In_Hid_Axon[i].get_weight( );
            float before = weight;
            //Serial.printf("weight[%02i]: %f\n", i, weight);
            float prv_wt = In_Hid_Axon[i].prev_weight;
            float delta = out->delta;
            prv_wt = learning_rate * input * delta + momentum * prv_wt;
            weight += prv_wt;
            In_Hid_Axon[i].set_weight(weight);
            In_Hid_Axon[i].prev_weight = prv_wt;
            Serial.printf("weight: %f | input: %f | prv: %f | delta: %f | learning_rate: %f | before: %f\n", weight, input, prv_wt, delta, learning_rate, before);
        }
        // ----------------------------------------------------------------
        for (int i = 0; i < hid_out_axon_count; i++) {
            Neuron *in  = Hid_Out_Axon[i].get_input_neuron( );
            Neuron *out = Hid_Out_Axon[i].get_output_neuron( );
            
            float input  = in->output_data_value;
            float weight = Hid_Out_Axon[i].get_weight( );
            
            float prv_wt = Hid_Out_Axon[i].prev_weight;
            float delta = out->delta;
            prv_wt = learning_rate * input * delta + momentum * prv_wt;
            weight += prv_wt;
            Hid_Out_Axon[i].set_weight(weight);
            In_Hid_Axon[i].prev_weight = prv_wt;
            //Serial.printf("weight: %f | Hid_Out_Axon[i].get_weight( ): %f\n", weight, Hid_Out_Axon[i].get_weight( ));
        }
        Serial.println();
        
    }
    
};*/

/*enum LAYER_TYPES {  UNINTIALIZED_LAYER,
                    INPUT_LAYER,
                    HIDDEN_LAYER,
                    OUTPUT_LAYER
};

class NeuralNetwork;
class Layer;
class Neuron;
class Axon;
//--------------------------------------------------------------------
class Neuron {
public:

private:
    Neuron ( ) : num_input_to_axons( 0 ), num_output_to_axons( 0 ) {
        Serial.println("Neuron");
    }
    
    void setLayerType ( LAYER_TYPES type ) {
        layer_type = type;
    }
    
    float Activation(float x) {
        float activatedValue = 1 / (1 + exp(-1 * x));
        return activatedValue;
    }
    
    void computeOutputValue( void ) {
        float accum = 0;
        switch ( layer_type ) {
            case INPUT_LAYER:
                neuron_output_value = neuron_input_value;
                break;
            case HIDDEN_LAYER:
                
                //for (int i = 0; i < num_axons; i++) { }
                break;
            case OUTPUT_LAYER:
                
                break;
                
            default:
                break;
        }
    }
    
    friend Layer;
    friend Axon;
    friend NeuralNetwork;
    LAYER_TYPES layer_type;
    float bias;
    float neuron_input_value;
    float neuron_output_value;
    float delta_error;
    uint16_t num_input_to_axons;
    uint16_t num_output_to_axons;
};
//--------------------------------------------------------------------
class Axon {
public:
    Axon ( void ) : input_neuron ( NULL), output_neuron ( NULL) {
        float rand = float(random(100)) / 100;
        weigth = 2.0f * ( rand - 0.5f ) * 0.5f;
    }
    
    void setInput ( Neuron *n ) {
        Serial.printf("input neuron: %p\n", n);
        input_neuron = n;
    }
    
    void setOutput ( Neuron *n ) {
        Serial.printf("output neuron: %p\n", n);
        output_neuron = n;
    }
private:
    void updateOutputNeuron ( void ) {
        float in = input_neuron->neuron_output_value;
        float out = output_neuron->neuron_input_value;
        out = in * weigth;
        output_neuron->neuron_input_value = out;
    }
    Neuron *input_neuron;
    Neuron *output_neuron;
    float weigth;
};
//--------------------------------------------------------------------
class Layer {
private:
    Neuron *nrons;
    uint16_t num_nrons;
    friend NeuralNetwork;
    Layer *next;
    
    uint16_t numNeurons ( void ) const {
        return num_nrons;
    }
    
    Neuron *getNeuron ( void ) const {
        return nrons;
    }
public:
    Layer ( uint16_t num_neurons, LAYER_TYPES l_type ) : num_nrons(num_neurons) {
        nrons = new Neuron[num_neurons];
        for (int i = 0; i < num_neurons; i++) {
            Serial.printf("neuron address: %p\n", &nrons[i]);
            nrons[i].setLayerType( l_type );
        }
    }
    
    ~Layer ( void ) {
        delete[] nrons;
    }
};
//--------------------------------------------------------------------
class NeuralNetwork {
public:
    template<class ...Tail>
    NeuralNetwork ( Layer &head, Tail&... tail ) {
        int inc = sizeof...( tail );
        if ( root_layer == NULL ) {
            root_layer = &head;
        }
        else {
            Layer *p;
            p = root_layer;
            for ( ; p->next; p = p->next );
            p->next = &head;
        }
        num_axons += head.num_nrons * prev_num_neurons;
        head.next = NULL;
        if ( inc <= 0 ) {
            axons = new Axon[num_axons];
            Layer *p, *u;
            p = root_layer;
            for ( ; p; p = p->next ) {
                if ( p->next == NULL ) break;
                u = p->next;
                Serial.printf("p: %p | u: %p\n\n", p, u);
                int idx = 0;
                Neuron *np = p->getNeuron( );
                Neuron *nu = u->getNeuron( );
                for ( int i = 0; i < p->numNeurons( ); i++ ) {
                    
                    np[i].num_output_to_axons = u->numNeurons( );
                    
                    
                    for ( int j = 0; j < u->numNeurons( ); j++ ) {
                        axons[idx].setInput( &np[i] );
                        axons[idx++].setOutput( &nu[j] );
                        Serial.println();
                    }
                    
                    Serial.println("--------------");
                }
                
            }
            Serial.printf("Num Axons: %i | num layers: %i\n", num_axons, num_layers);
        }
        
        prev_num_neurons = head.num_nrons;
        num_layers++;
        NeuralNetwork( tail... );
    }
    
    
    ~NeuralNetwork ( void ) {
        delete[] axons;
    }
private:
    static Layer *root_layer;
    static uint16_t num_axons;
    static uint16_t prev_num_neurons;
    static uint8_t num_layers;
    Axon *axons;
    
    NeuralNetwork ( void ) {
        Serial.println("Neural Network Constructor");
    }
};

/*
class Neuron;
class Layer;
class NeuralNetwork;

enum LAYER_TYPES {
    UNINTIALIZED_LAYER,
    INPUT_LAYER,
    HIDDEN_LAYER,
    OUTPUT_LAYER
};

#define AllocateNeurons(num) ({   \
    static Neuron n[num];               \
    Neuron::init( n, num );             \
    n;                                  \
})

class Axon {
public:
    Axon ( void ) {
        Serial.println("Axon Constructor");
        //randomiseWeight();
    }
private:
    float axonEntry;
    float weight;
    float axonExit;
    //Function to set the weight of this connection
    void setWeight(float tempWeight) {
        //weight = tempWeight;
    }
    
    //Function to randomise the weight of this connection
    void randomiseWeight() {
        //setWeight(random(2) - 1);
    }
    
    //Function to calculate and store the output of this Connection
    float calcAxonExit(float tempInput) {
        //axonEntry = tempInput;
        //axonExit = connEntry * weight;
        //return connExit;
    }
};

class Neuron {
public:
    Neuron ( void ) : neuron_cnt( 0 ), prev_neuron_cnt( 0 ), l_type( UNINTIALIZED_LAYER ) {
        Serial.printf("Neuron Constructor: %i\n", neuron_count++);
        if ( root_neuron == NULL ) {
            root_neuron = this;
            
        }
        else {
            Neuron *p;
            p = root_neuron;
            for ( ; p->next; p = p->next );
            p->next = this;
        }
        next = NULL;
    }
    
    Neuron ( LAYER_TYPES t ) {
        
    }
    
    static void init( Neuron *nron, uint8_t num ) {
        Serial.printf("Neuron Address: %p | number of: %i | count: %i\n", nron, num, neuron_count);
        if (nron->l_type == HIDDEN_LAYER) {
            Serial.println("Yes");
        }
        nron->neuron_cnt = neuron_count;
        neuron_count = 0;
    }
    static Neuron *root_neuron;
    Neuron *next;
    static uint16_t neuron_count;
    uint16_t neuron_cnt;
    uint16_t prev_neuron_cnt;
    LAYER_TYPES l_type;
private:
};

class Layer {
private:
    
    Neuron *neurons;
public:
    Layer ( void ) : num_neurons( 0 ) {
        Serial.println("Layer Constructor");
    }
    
    Layer ( uint16_t n ) : num_neurons( 0 ) {
        num_neurons = n;
        Serial.println("Layer Constructor 2");
    }
    
    void addNeurons( Neuron *n ) {
        Serial.printf("Add Neurons: %p | Neuron Count: %i\n", n, n->neuron_cnt);
        Neuron *p;
        for (p = n; p; p = p->next ) {
            switch ( l_type ) {
                case INPUT_LAYER:
                    p->l_type = INPUT_LAYER;
                    break;
                case HIDDEN_LAYER:
                    p->l_type = HIDDEN_LAYER;
                    break;
                case OUTPUT_LAYER:
                    p->l_type = OUTPUT_LAYER;
                    break;
                    
                default:
                    break;
            }
            
            Serial.printf("p neuron: %p | \n", p);
        }
        Serial.println();
        neurons = n;
    }
    static uint16_t prev_neuron_count;
    LAYER_TYPES l_type;
    Layer *next;
    uint16_t num_neurons;
};

class NeuralNetwork {
public:
    template<class ...Tail>
    NeuralNetwork( Layer &head, Tail&... tail ) {

        int i = sizeof...( tail );
        
        Serial.printf("Network Template Constructor: %i | prev neurons: %i | head.num_neurons: %i | sum: %i | axon: %i\n", i, prev_num_neurons, head.num_neurons,prev_num_neurons * head.num_neurons, num_axons);
        
        if ( root_layer == NULL ) {
            root_layer = &head;
            head.l_type = INPUT_LAYER;
        }
        else {
            Layer *p;
            p = root_layer;
            for ( ; p->next; p = p->next );
            p->next = &head;
            head.l_type = HIDDEN_LAYER;
        }
        
        num_axons += head.num_neurons * prev_num_neurons;
        
        head.next = NULL;
        
        if ( i <= 0 ) {
            head.l_type = OUTPUT_LAYER;
        }
        
        prev_num_neurons = head.num_neurons;
        NeuralNetwork( tail... );
    }
private:
    NeuralNetwork( void ) {
        Serial.printf("Network Constructor | Num of Axons: %i\n", num_axons);
    }
    static Layer *root_layer;
    static uint16_t num_axons;
    static uint16_t prev_num_neurons;
};*/
#endif /* NeuralNetwork_hpp */
