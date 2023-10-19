using Flux, JLD2


function load_state!(flux_model, model_state_file_path::String)
    model_state = JLD2.load(model_state_file_path, "model_state");
    Flux.loadmodel!(flux_model, model_state);
end

include("RDNDenoiser.jl")
include("Pseudo3D.jl")