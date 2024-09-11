bag = rosbag("C:\Users\35840\Documents\20211217_204044.bag");
topics = bag.AvailableTopics;

% Get only the option topics
depth_sensor_option_topics = topics(14:66,:).Properties.RowNames;
rgb_sensor_option_topics = topics(73:105,:).Properties.RowNames;
imu_sensor_option_topics = topics(117:122,:).Properties.RowNames;

% Put options data in cells
depth_options = options_to_cell(bag, depth_sensor_option_topics)
rgb_options = options_to_cell(bag, rgb_sensor_option_topics)
imu_options = options_to_cell(bag, imu_sensor_option_topics)

cell_to_file(depth_options, 'depth_options.txt')
cell_to_file(rgb_options, 'rgb_options.txt')
cell_to_file(imu_options, 'imu_options.txt')
function cell_to_file(cell, file_name)
    file = fopen(file_name,'w')
    for index = 1 : numel(cell) / 2
        disp(index)
        if mod(index,2) == 0
            fprintf(file,'%s, %f\n',cell{index,1},cell{index,2});
        else
            fprintf(file,'%s, %s\n',cell{index,1},cell{index,2});
        end
    end
    fclose(file);
end

function options = options_to_cell(bag, topics)
    for index = 1 : numel(topics)
        m = readMessages(select(bag,'Topic',topics{index}),'DataFormat','struct');
        options{index,1} = topics{index};
        if mod(index,2) == 0
            options{index,2} = m{1,1}.Data;
        else
            options{index,2} = m{1,1}.Data;
        end
    end
end