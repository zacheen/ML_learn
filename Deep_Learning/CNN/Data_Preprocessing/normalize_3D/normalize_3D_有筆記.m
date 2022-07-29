clear all
addpath('C:\TuanShu\MATLAB\TSLib');



If_AutoMin=1; % 此變數後面不會再出現了
If_Max_Correction=1; % 此變數後面不會再出現了
% If align glass
If_Cut_Front=0; % 前面幾張 frame 丟棄
If_Adv_Normalized_By_Max=1; % 此變數後面不會再出現了

Cut_Tail=0; % 此變數後面不會再出現了
% If Auto-brightness
If_TopNPerc=0; % 此變數後面不會再出現了

% File Parameters
Parent_Parent_Folder_Path=['D:\210806_Anny Data\vue20210805142119\patient\new\'];
%File_Name=['20201115_111230C.bin'];
List=dir('D:\210806_Anny Data\vue20210805142119\patient\new\\\**\\*.bin');


%% Scan Parent Folder

dirFlags = [List.isdir];
% Extract only those that are directories.
file_list = List(~dirFlags);


for TTT=1:length(file_list)
    Data_Save_Folder_Path=[file_list(TTT).folder  '\\Result'];
    if exist(Data_Save_Folder_Path)==0
        mkdir(Data_Save_Folder_Path);
    end
    
    
    % Lateral Blur on Second Image
            %2.2
    File_Name=file_list(TTT).name;

    X_ROI=[1 1024];
    Y_ROI=[1 1024];  %715
    % Image Info
    Image_Height=1024;%after PostAVE Now;
    Image_Width=1024;
    Byte_Depth=4;       %4 for float

    File_Size_Byte=file_list(TTT).bytes;
    Estimated_Total_Frame=File_Size_Byte/(Image_Height*Image_Width*Byte_Depth);
	% Estimated_Total_Frame 每次只讀取 100 個 frame，應該是怕檔案太大
	
    if Estimated_Total_Frame <100 % 這裡有點奇怪 這樣最後 100 層不是會沒有做到
        continue;
    end
    
    file_path=[file_list(TTT).folder '\\' file_list(TTT).name];

    fin=fopen(file_path);

    Image_Stack=zeros(Image_Height,Image_Width,Estimated_Total_Frame);
    for p=1:Estimated_Total_Frame
        Byte_Skip=Image_Height*Image_Width*Byte_Depth*(p-1);
        fseek(fin, Byte_Skip, 'bof');
        Image_Stack(:,:,p)=fread(fin,[Image_Height,Image_Width],'float32',0,'l');
        disp(p);
    end
    fclose('all');

    %%
    if If_Cut_Front > 0
        Image_Stack=Image_Stack(:,:,If_Cut_Front:end);
    end

    %% Blur on E
    Image_Stack_Blur=Image_Stack;
    Sigma=1;
    for p=1:size(Image_Stack,3)
        Image_Stack_Blur(:,:,p) = imgaussfilt(Image_Stack_Blur(:,:,p),Sigma);
    end


    %% Axial Histogram adj
    Image_Stack_Blur_AHist_Corr=Image_Stack_Blur;
    % tag 1 XY 前處理 (參數)
    Adj1_A=1;
    Adj2_A=1;
    % 跟lateral之不同是, 不能直接將不同深度影像之Histogram調到相同
    % 只能blur
    Buffer_Length=10;
    Cmin_Buffer_Array=zeros(Buffer_Length,1);
    Cmax_Buffer_Array=zeros(Buffer_Length,1);

    % tag 2 XY 前處理 (根據前10張的亮度 進行亮度調整)
    for p=1:size(Image_Stack_Blur,3)

        Current_Image=Image_Stack_Blur(:,:,p);
        ROI_Image=Current_Image(50:(end-50),50:(end-50));

        Sort_descend=sort(ROI_Image(:),'descend');
        Sort_ascend=sort(ROI_Image(:),'ascend');

        Cmin_Now=Sort_ascend(round(length(Sort_ascend)*Adj1_A/100));
        Cmax_Now=Sort_descend(round(length(Sort_descend)*Adj2_A/100))*1;

        Cmin_Buffer_Array(rem(p-1,Buffer_Length)+1)=Cmin_Now;
        Cmax_Buffer_Array(rem(p-1,Buffer_Length)+1)=Cmax_Now;

        Cmin_Buffered=mean(nonzeros(Cmin_Buffer_Array));
        Cmax_Buffered=mean(nonzeros(Cmax_Buffer_Array));

        Image_Temp=(Current_Image-Cmin_Now)./(Cmax_Now-Cmin_Now);

        Image_Adjed=Image_Temp*(Cmax_Buffered-Cmin_Buffered)+Cmin_Buffered;
        Image_Adjed(Image_Adjed<0)=0;

        Image_Stack_Blur_AHist_Corr(:,:,p)=Image_Adjed;
        disp(p);
    end


    %% Generate B-scan Stack - 1

    % tag 3 XZ 前處理
    B_Stack_1=permute(Image_Stack_Blur_AHist_Corr,[3 1 2]); # [ZYX] -> [XZY]
    %%
    Total_Number_of_Frame=size(B_Stack_1,3); %+size(B_Stack_2,3);

    Processed_B_Stack=zeros(size(B_Stack_1,1),size(B_Stack_1,2),Total_Number_of_Frame);

    for p=1:Total_Number_of_Frame
        Image_Original=B_Stack_1(:,:,p);

        Image_Log=log(Image_Original);

        Image_Log(isinf(Image_Log))=NaN;

        Image_Log_Sort_1D=sort(Image_Log(:),'ascend');
        Min_by_Sort=mean(Image_Log_Sort_1D(1:round(length(Image_Log_Sort_1D)/10)),'omitnan');

        Image_Log=Image_Log-Min_by_Sort;
        Image_Log(Image_Log<0)=0;
        Image_Log(isnan(Image_Log))=0;


        % Three Step Pyramid
        % 1st: high resolution
        % 2nd: mid resolution that we care (i.e. collegen)
        % 3rd: low resolution 


        Median_1=1;
        Median_2=1;
        Blur_1=5;
        Blur_2=20;
        Exp_1=1;
        Exp_2=1.5;
        Exp_3=0.45;


        C_max_Def=2;%3.9  %4.5
        C_min_Def=0.6;%2.1  %2.7

        Nperc=0.03;
        Cmin_Ratio=0.48;

        Cmax_AB=0.99;
        Cmin_AB=0.42;

        # tag 4 XZ 前處理(開始處理) 
        Image_Blur_1=imgaussfilt(Image_Log,Blur_1);
        Image_Blur_2=imgaussfilt(Image_Log,Blur_2);


        Image_1st_Backup=Image_Log-Image_Blur_1;
        Image_1st_Backup_med2=medfilt2(Image_1st_Backup,[Median_1 Median_1]);
        Image_1st=Image_1st_Backup_med2.^Exp_1;
        Image_1st=Image_1st./max(Image_1st(:)).*max(Image_1st_Backup_med2(:));


        Image_2nd_Backup=Image_Blur_1./Image_Blur_2;
        Image_2nd_Backup2=medfilt2(Image_2nd_Backup,[Median_2 Median_2]);
        Image_2nd=Image_2nd_Backup2.^Exp_2;
        Image_2nd=Image_2nd./max(Image_2nd(:)).*max(Image_2nd_Backup2(:));

        Image_3rd_Backup=Image_Blur_2;
        Image_3rd=Image_3rd_Backup.^Exp_3;
        Image_3rd=Image_3rd./max(Image_3rd(:)).*max(Image_3rd_Backup(:));

        Image_Rec=Image_3rd.*Image_2nd+Image_1st; %.*Image_Map_1st;
   


        Processed_B_Stack(:,:,p)=Image_Rec;
        %imagesc(Image_Rec)
        disp(p);
    end

    % tag 5 最後處理
    Adj1=1;
    Adj2=1;%0.3

    %%
    Histogram_Corr_Image_Stack=Processed_B_Stack;

    for p=1:size(Processed_B_Stack,3)

        Current_Image=Processed_B_Stack(:,:,p);
        ROI_Image=Current_Image(30:end,100:925);

        Sort_descend=sort(ROI_Image(:),'descend');
        Sort_ascend=sort(ROI_Image(:),'ascend');

        Cmin=Sort_ascend(round(length(Sort_ascend)*Adj1/100));
        Cmax=Sort_descend(round(length(Sort_descend)*Adj2/100))*1.15;
        Image_Norm=(Current_Image-Cmin)./(Cmax-Cmin);
        Image_Norm(Image_Norm>1)=1;
        Image_Norm(Image_Norm<0)=0;
        Histogram_Corr_Image_Stack(:,:,p)=Image_Norm;
        disp(p);
    end

    Histogram_Corr_Image_Stack=permute(Histogram_Corr_Image_Stack,[2 3 1]); # [XZY] -> [ZYX]
    fid = fopen([Data_Save_Folder_Path '\' File_Name  '_3D_DFcorr_not enhancing depth.raw'], 'w+');
    fwrite(fid, Histogram_Corr_Image_Stack, 'single');
    fclose(fid);
end