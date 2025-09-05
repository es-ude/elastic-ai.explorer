// Auto-generated resolver ops
// Generated from: lstm_model.tflite
    resolver->AddReshape();
    resolver->AddFullyConnected();
    resolver->AddSlice();
    resolver->AddAdd();
    resolver->AddLogistic();
    resolver->AddTanh();
    resolver->AddMul();
    // Add mapping for BROADCAST_TO
    resolver->AddConcatenation();
