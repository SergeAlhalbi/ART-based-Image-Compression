%==========================================================================
% Project 5
%==========================================================================

%% Load image
name{1} = 'cameraman.tif';
name{2} = 'peppers.png';
name{3} = 'mandrill.jpg';

for im = 1 : length(name)
    % Show oringinal image
    o_image = imread(name{im});
    figure;
    subplot(4,2,1);
    imshow(o_image);
    formatSpec = 'Original image %d %s';
    str = sprintf(formatSpec,im,name{im});
    title(str);
    
    % Determine if the image is rgb or gray
    if length(size(o_image))>2
        image = im2double(rgb2gray(o_image));
    else
        image = im2double(o_image);
    end
    
    % Show gray image
    subplot(4,2,2);
    imshow(image);
    formatSpec = 'Gray scale image %g %s';
    str = sprintf(formatSpec,im,name{im});
    title(str);
    
    %% Set block parameters
    block_size = 4; % 4x4 blocks
    
    [I_row, I_col] = size(image);
    
    block_row = I_row/block_size;
    block_col = I_col/block_size;
    
    block_number = block_row * block_col;
    
    %% Create the compression image
    compression_image = zeros(block_size*block_size,block_number);
    block = zeros(block_size,block_size,block_number);
    for i = 1 : block_row
        for j = 1: block_col
            number = block_col*(i-1)+ j;
            block(:,:,number) = image(block_size*(i-1)+1 : block_size*i , block_size*(j-1)+1 : block_size*j);
            compression_image(:,number) = reshape(block(:,:,number),[],1);
        end
    end
    
    %% Compression
    
    % Complement input
    X = [compression_image ; 1 - compression_image];
    
    % Set choice parameter
    aleph = 0.001;
    % Set vigilance parameter
    p_value = [0.85,0.9,0.93];
    % Set learning rate parameter
    beta_value = 0.2;
    b = 1;
    
    for p = 1:length(p_value)
        % Initial weights
        w = zeros(size(X,1),1);
        
%         New_output_label = 0;
        
%         flag = 0;
%         iteration = 0;

        % Initial labels
        output_class = zeros(1,1);

%         while (flag == 0) % (System stabalization will take a lot of time)

            for i = 1 : block_number
            
                [row, col_w] = size(w);
            
                X_choice = repmat(X(:,i),[1,col_w]);
            
            
                % Compute the choice function
                numerator_CF_train = sum(min(w,X_choice));
            
                denominator_CF_train = aleph + sum(w);
            
                CF = numerator_CF_train ./ denominator_CF_train;
            
                [CF_J, J] = max(CF);
            
                % Compute the match function or the category choice
                numerator_MF_train = sum(min(w(:,J),X(:,i)));
            
                denominator_MF_train = sum(X(:,i));
            
                MF =  numerator_MF_train / denominator_MF_train;
            
                j = 0; % No resets initially
            
                while( MF < p_value(p) )

                        % Create a new node if all previous nodes fail
                        % (All MFs are zeros)

                    if j == col_w
                        w = horzcat(w,X(:,i));
                    
                        [row, col] = size(w);
                    
                        numerator_CF_train = sum(min(w(:,col),X(:,i)));
                    
                        denominator_CF_train = aleph + sum(w(:,col));
                    
                        CF(col) = numerator_CF_train / denominator_CF_train;
                    
                    end

                    % Reset the winner node (Some steps are additional but don't affect the output)

                    CF(J) = 0;
                
                    [CF_J, J] = max(CF);
                
                    numerator_MF_train = sum(min(w(:,J),X(:,i)));
                
                    denominator_MF_train = sum(X(:,i));
                
                    MF =  numerator_MF_train / denominator_MF_train;
                
                    j = j+1;
                
                end
            
                % Update weights
                w(:,J) = beta_value(b) * min( w(:,J) , X(:,i) ) + (1-beta_value(b)) * w(:,J);
            
                % Update labels
                [cl_row,cl_col] = size(output_class);
                if j == col
                    output_class(1,cl_col+1) = i;
                else
                    output_class(cl_row+1,J) = i;
                end
            end
        
            % Compute output label
            [cl_row,cl_col] = size(output_class);
            output_label = 0;
            for k = 2: cl_col
                group = output_class(:,k);
                label = group(group~=0);
                output_label(1:size(label,1),k-1) = label;
            end
        
%             % Repeat some times untill no change
%             if size(New_output_label) == size(output_label)
%                 if New_output_label == output_label
%                     flag = 1;
%                 end
%             end
        
%             New_output_label = output_label;
        
%             iteration = iteration+1;
%         end
        
        block_code = output_label(1,1:end);
        code_book = block(:,:,block_code);
        compressed_image = compression_image(:,block_code);
        
        block_code_number = zeros(1,number);

        for i = 1 : number
            
            [~,block_code_number(i)] = find(output_label == i);
            
        end

        % Or
%         for i = 1 : number
%             
%             block_code_number(i) = ceil(find(output_label == i,2)/size(output_label,1));
%             
%         end

        %% Reconstruction
        reconstruct_block = zeros(block_size,block_size,number);
        for i = 1 : number
            reconstruct_block(:,:,i) = code_book(:,:,block_code_number(i));
        end
        
        reconstruct_image = zeros(size(image));
        
        for i = 1 : block_row
            for j = 1: block_col
                number = block_col*(i-1)+ j;
                
                reconstruct_image(block_size*(i-1)+1 : block_size*i , block_size*(j-1)+1 : block_size*j)...
                    = reconstruct_block(:,:,number);
                
            end
        end
        
        %% Low pass filter
        
        radius = 1;
        J1 = fspecial('disk', radius);
        low_pass_im = imfilter(reconstruct_image,J1,'replicate');
        
        %% Compression ratio
        MSE_error(im,p) = sum(sum( (image - low_pass_im).^2 ))/(I_row * I_col);
        
        PSNR(im,p) = 10 * log10(1/MSE_error(im,p));
        
        CR_N = size(image,1)*size(image,2);
        CR_D = (block_size^2*+1) * size(code_book,3);
        
        CR(im,p) = CR_N/CR_D;
        
        %% Plot
        subplot(4,2,2*p+1);
        imshow(low_pass_im);
        str = sprintf('Learning rate %g, Vigilance %g'...
            ,beta_value,p_value(p));
        title(str);
        
        
    end
    subplot(4,2,4);
    plot(p_value, MSE_error(im,:),'--gs','LineWidth',2,'MarkerSize',10,...
        'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5]);
    xlabel('vigilance parameter');
    ylabel('MSE error');
    title('Mean Square Error');
    
    subplot(4,2,6);
    plot(p_value, PSNR(im,:),'--bs','LineWidth',2,'MarkerSize',10,...
        'MarkerEdgeColor','m','MarkerFaceColor',[0.5,0.5,0.5]);
    xlabel('vigilance parameter');
    ylabel('PSNR');
    title('Peak Signal to Noise Ratio');
    
    subplot(4,2,8);
    plot(p_value, CR(im,:),'--rs','LineWidth',2,...
        'MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor',[0.5,0.5,0.5]);
    xlabel('vigilance parameter');
    ylabel('CR');
    title('Compression ratio');
    
end
%==========================================================================