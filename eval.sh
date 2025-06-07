# set -e
# configs=configs/Dynamic

# # For Bonn 
# for config in "$configs"/Bonn/*
# do
#     if [[ $config == *"_dynamic"* ]]; then
#         continue
#     fi
#     version="${config##*/}"
#     version="${version%.*}"
#     mkdir -p "output/Bonn/${version}"
#     echo "{ time python run.py "$config"; } 2>&1 | tee "output/Bonn/${version}/log.txt""

# done

# # For TUM
# for config in "$configs"/TUM_RGBD/*
# do
#     if [[ $config == *"_dynamic"* ]]; then
#         continue
#     fi
#     version="${config##*/}"
#     version="${version%.*}"
#     mkdir -p "output/TUM_RGBD/${version}"
#     echo "{ time python run.py "$config"; } 2>&1 | tee "output/TUM_RGBD/${version}/log.txt""
# done

# # For Wild_SLAM_Mocap
# for config in "$configs"/Wild_SLAM_Mocap/*
# do
#     if [[ $config == *"_slam_mocap"* ]]; then
#         continue
#     fi
#     if [[ $config == *"_demo"* ]]; then
#         continue
#     fi
#     version="${config##*/}"
#     version="${version%.*}"
#     mkdir -p "output/Wild_SLAM_Mocap/${version}"
#     echo "{ time python run.py "$config"; } 2>&1 | tee "output/Wild_SLAM_Mocap/${version}/log.txt""
# done



{ time python run.py configs/Dynamic/Bonn/bonn_balloon.yaml; } 2>&1 | tee output/Bonn/bonn_balloon/log.txt
{ time python run.py configs/Dynamic/Bonn/bonn_balloon2.yaml; } 2>&1 | tee output/Bonn/bonn_balloon2/log.txt
{ time python run.py configs/Dynamic/Bonn/bonn_crowd.yaml; } 2>&1 | tee output/Bonn/bonn_crowd/log.txt
{ time python run.py configs/Dynamic/Bonn/bonn_crowd2.yaml; } 2>&1 | tee output/Bonn/bonn_crowd2/log.txt
{ time python run.py configs/Dynamic/Bonn/bonn_moving_nonobstructing_box.yaml; } 2>&1 | tee output/Bonn/bonn_moving_nonobstructing_box/log.txt
{ time python run.py configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2.yaml; } 2>&1 | tee output/Bonn/bonn_moving_nonobstructing_box2/log.txt
{ time python run.py configs/Dynamic/Bonn/bonn_person_tracking.yaml; } 2>&1 | tee output/Bonn/bonn_person_tracking/log.txt
{ time python run.py configs/Dynamic/Bonn/bonn_person_tracking2.yaml; } 2>&1 | tee output/Bonn/bonn_person_tracking2/log.txt

{ time python run.py configs/Dynamic/TUM_RGBD/freiburg3_walking_halfsphere.yaml; } 2>&1 | tee output/TUM_RGBD/freiburg3_walking_halfsphere/log.txt
{ time python run.py configs/Dynamic/TUM_RGBD/freiburg3_walking_halfsphere_static.yaml; } 2>&1 | tee output/TUM_RGBD/freiburg3_walking_halfsphere_static/log.txt
{ time python run.py configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml; } 2>&1 | tee output/TUM_RGBD/freiburg3_walking_rpy/log.txt
{ time python run.py configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml; } 2>&1 | tee output/TUM_RGBD/freiburg3_walking_xyz/log.txt

{ time python run.py configs/Dynamic/Wild_SLAM_Mocap/ANYmal1.yaml; } 2>&1 | tee output/Wild_SLAM_Mocap/ANYmal1/log.txt
{ time python run.py configs/Dynamic/Wild_SLAM_Mocap/table_tracking1.yaml; } 2>&1 | tee output/Wild_SLAM_Mocap/table_tracking1/log.txt
{ time python run.py configs/Dynamic/Wild_SLAM_Mocap/umbrella.yaml; } 2>&1 | tee output/Wild_SLAM_Mocap/umbrella/log.txt