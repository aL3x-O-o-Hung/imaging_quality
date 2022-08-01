import pydicom
import os
import csv
import shutil
import numpy as np
def convert_num_to_string(num,length=4):
    s=str(num)
    while len(s)!=length:
        s='0'+s
    return s

def parse_date(date):
    date=date[1:]
    first=date.find('/')
    second=date.find('/',first+1)
    return '20'+date[second+1:]+convert_num_to_string(date[:first],length=2)+convert_num_to_string(date[first+1:second],length=2)

def parse_time(time):
    time=time[1:]
    end=time.find('M')+1
    time=time[:end]
    num_index=time.find(' ')
    if time[-2]=='A':
        offset=0
    elif time[-2]=='P':
        offset=12
    hour=int(time[:num_index-2])+offset
    if hour>=24:
        hour-=12
    minute=time[num_index-2:num_index]
    return convert_num_to_string(hour,length=2)+convert_num_to_string(minute,length=2)


def find_mapping(folder='../data/20160131/',dic={}):
    p1=folder+'t2_tse_tra_320_p2/'
    p2=folder+'ep2dadvdiff3Scan4bval_spair_std_ADC/'
    p3=folder+'Localizers/'
    p1_dirs={}
    p2_dirs={}
    p3_dirs={}
    dirs=os.listdir(p1)
    for file in dirs:
        if file!='.DS_Store':
            if file[3:7] not in p1_dirs:
                ds=pydicom.read_file(p1+'IM-'+file[3:7]+'-0001.dcm')
                p1_dirs[file[3:7]]=[ds.StudyInstanceUID,float(ds.ImagePositionPatient[2])]
                #a=ds.pixel_array
                #print(a.shape)
    dirs=os.listdir(p2)
    for file in dirs:
        if file!='.DS_Store':
            if file[3:7] not in p2_dirs:
                ds=pydicom.read_file(p2+'IM-'+file[3:7]+'-0001.dcm')
                p2_dirs[file[3:7]]=[ds.StudyInstanceUID,float(ds.ImagePositionPatient[2])]
                #a=ds.pixel_array
                #print(a.shape)
    dirs=os.listdir(p3)
    for file in dirs:
        if file!='.DS_Store':
            if file[3:7] not in p3_dirs:
                ds=pydicom.read_file(p3+'IM-'+file[3:7]+'-0001.dcm')
                if ds.AcquisitionDate in dic.keys():
                    if ds.AcquisitionTime[0:4] in dic[ds.AcquisitionDate].keys():
                        temp=dic[ds.AcquisitionDate][ds.AcquisitionTime[0:4]]
                        p3_dirs[file[3:7]]=[ds.StudyInstanceUID,temp[0],temp[1],temp[2]]
    out=[]
    for k1 in p1_dirs.keys():
        for k2 in p2_dirs.keys():
            if p1_dirs[k1][0]==p2_dirs[k2][0] and np.abs(p1_dirs[k1][1]-p2_dirs[k2][1])<=1:
                temp=[k1,k2]
                for k3 in p3_dirs.keys():
                    if p1_dirs[k1][0]==p3_dirs[k3][0]:
                        temp.append(p3_dirs[k3][1])
                        temp.append(p3_dirs[k3][2])
                        temp.append(p3_dirs[k3][3])
                out.append(temp)
    real_out=[]
    for o in out:
        if len(o)==2:
            real_out.append([p1+'IM-'+o[0]+'-',p2+'IM-'+o[1]+'-'])
        else:
            real_out.append([p1+'IM-'+o[0]+'-',p2+'IM-'+o[1]+'-',[o[2],o[3],o[4]]])
    #print(real_out)
    return real_out

def read_csv(path='../data/Summary_Results.csv'):
    i=-1
    dic={}
    with open(path,newline='',encoding='utf-8') as csvfile:
        reader=csv.reader(csvfile,delimiter=',',quotechar='|')
        for row in reader:
            i+=1
            if i==0:
                continue
            elif len(row)!=19:
                break
            temp=parse_date(row[0])
            if temp not in dic.keys():
                dic[temp]={}
            dic[temp][parse_time(row[1])]=[float(row[3]),float(row[4]),float(row[5])]
    return dic

def resave(in_folder='../data/prostateDWI/DICOM/',out_folder='../data/ProstateImageQuality1/',csv_path='../data/prostateDWI/Summary_Results.csv'):
    dirs=os.listdir(in_folder)
    l=[]
    dic=read_csv(csv_path)
    for dir in dirs:
        if dir=='.DS_Store':
            continue
        temp=find_mapping(in_folder+dir+'/',dic)
        l.extend(temp)
    no_score_count=-1
    score_count=-1
    if not os.path.exists(out_folder+'no_score/'):
        os.mkdir(out_folder+'no_score/')
    if not os.path.exists(out_folder+'score/'):
        os.mkdir(out_folder+'score/')
    for item in l:
        if len(item)==2:
            no_score_count+=1
            temp_out=out_folder+'no_score/'+str(no_score_count)+'/'
        elif len(item)==3:
            print(score_count)
            score_count+=1
            temp_out=out_folder+'score/'+str(score_count)+'/'
        if not os.path.exists(temp_out):
            os.mkdir(temp_out)
        if not os.path.exists(temp_out+'T2/'):
            os.mkdir(temp_out+'T2/')
        if not os.path.exists(temp_out+'ADC/'):
            os.mkdir(temp_out+'ADC/')
        for i in range(1,21):
            shutil.copyfile(item[0]+convert_num_to_string(i)+'.dcm',temp_out+'T2/'+str(i)+'.dcm')
            shutil.copyfile(item[1]+convert_num_to_string(i)+'.dcm',temp_out+'ADC/'+str(i)+'.dcm')
        if len(item)==3:
            np.save(temp_out+'score.npy',np.array(item[2]))


def clean_up_dataset(old_folder='../data/ProstateImageQuality1_/',new_folder='../data/ProstateImageQuality1/'):
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    for mode in ['score/','no_score/']:
        i=0
        k=0
        if not os.path.exists(new_folder+mode):
            os.mkdir(new_folder+mode)
        while os.path.exists(old_folder+mode+str(i)+'/'):
            flag=True
            for j in range(1,21):
                try:
                    tempa=pydicom.dcmread(old_folder+mode+str(i)+'/T2/'+str(j)+'.dcm')
                    tempa=tempa.pixel_array
                    tempa=pydicom.dcmread(old_folder+mode+str(i)+'/ADCresampled/'+str(j)+'.dcm')
                    tempa=tempa.pixel_array
                except:
                    flag=False
                    break
            if flag:
                if not os.path.exists(new_folder+mode+str(k)+'/'):
                    os.mkdir(new_folder+mode+str(k)+'/')
                if not os.path.exists(new_folder+mode+str(k)+'/NII/'):
                    os.mkdir(new_folder+mode+str(k)+'/NII/')
                for m in ['T2/','ADC/','ADCresampled/']:
                    shutil.copyfile(old_folder+mode+str(i)+'/NII/'+m[:-1]+'.nii',new_folder+mode+str(k)+'/NII/'+m[:-1]+'.nii')
                    if not os.path.exists(new_folder+mode+str(k)+'/'+m):
                        os.mkdir(new_folder+mode+str(k)+'/'+m)
                    for j in range(1,21):
                        shutil.copyfile(old_folder+mode+str(i)+'/'+m+str(j)+'.dcm',new_folder+mode+str(k)+'/'+m+str(j)+'.dcm')

                if mode=='score/':
                    shutil.copyfile(old_folder+mode+str(i)+'/score.npy',new_folder+mode+str(k)+'/score.npy')
                k+=1
            i+=1





#dic=read_csv()
#find_mapping(csv_dic=dic)
#resave()
clean_up_dataset()