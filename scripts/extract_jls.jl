using IceFloeTracker 
using Images
using FileIO
using CSV
using IceFloeTracker: deserialize, DataFrames, float64, mosaicview, Gray

# making path to where the data is stored - customize this to what you need
algorithm = "ift_pipeline_default"
region = "beaufort_sea"
case = "beaufort_sea-100km_by_100km-20040428-20040429"
results_loc = joinpath("../data/ift_data/", algorithm, "ift_results", region, case);

im_save_loc = "../data/images/"
df_save_loc = "../data/tables/"

# landmasks
landmasks = deserialize(joinpath(results_loc, "landmasks/generated_landmask.jls"));

# preprocess output
# not all of this is used in this script, this is just in case you want this stuff
fnames = deserialize(joinpath(results_loc, "preprocess/filenames.jls")); # fnames[1] is the truecolor images
floe_props = deserialize(joinpath(results_loc, "preprocess/floe_props.jls"));
passtimes = deserialize(joinpath(results_loc, "preprocess/passtimes.jls"));
segmented_floes = deserialize(joinpath(results_loc, "preprocess/segmented_floes.jls"));
timedeltas = deserialize(joinpath(results_loc, "preprocess/timedeltas.jls"));

# tracker output
tracked_floes = deserialize(joinpath(results_loc, "tracker/labeled_floes.jls"));

# first output the individual floe property tables
for (f, df_props, im) in zip(fnames[1], floe_props, segmented_floes)
    CSV.write(joinpath(df_save_loc, replace(f, "tiff" => "props.csv")) , df_props)
    Images.save(joinpath(im_save_loc, f) , Gray.(Int.(im)))
end

# this is for a singled tracked floe dataframe
# for longer runs, you'd want to set up a for loop similar to the one above
CSV.write(joinpath(df_save_loc, "tracked.csv") , tracked_floes)