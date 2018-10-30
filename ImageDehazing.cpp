
 //
 //  main.cpp
 //  OpenCVExample
 //
 //  Created by Jiangnan Li on 10/18/16.
 //  Copyright © 2016 Jiangnan Li. All rights reserved.
 //
 
#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <stack>

#define Pi 3.1415926
#define Emin  0.0000001

using namespace std;
using namespace cv;

class Pixel
{
private:
    int red, blue, green;
    int gray;
    int row, col;
public:
    Pixel(int i, int j, int k, int a, int b)
    { red = i; green = j; blue = k; row = a; col = b; gray = NULL;}
    Pixel(int i, int j, int k)
    { gray = i; row = j; col = k; red = NULL; blue = NULL; green = NULL;}
    Pixel(int i) {gray = i; red = NULL; blue = NULL; green = NULL; row = NULL; col = NULL;}
    Pixel() {gray = NULL; red = NULL; blue = NULL; green = NULL; row = NULL; col = NULL;}

    void set_RGB (int i, int j, int k)
    { red = i; green = j; blue = k; }
    void set_gray (int i) { gray = i; }
    int get_gray() {return gray;}
    int get_row() {return row;}
    int get_col() {return col;}
    int get_brightness() {return (red + blue + green);}
    int get_red () {return red;}
    int get_blue() {return blue;}
    int get_green() {return green;}
};

int Findminimum (int value_1, int value_2, int value_3)
{
    int minimum;
    minimum = (value_1 < value_2 )? value_1 : value_2 ;
    minimum = (minimum < value_3 )? minimum : value_3 ;
    return minimum;
}

float FindminimumFloat (float value_1, float value_2, float value_3)
{
    float minimum;
    minimum = (value_1 < value_2 )? value_1 : value_2 ;
    minimum = (minimum < value_3 )? minimum : value_3 ;
    return  minimum;
}

void Find_DarkChannel (Mat * BGR, Mat * DarkChannel)
{
    int Findminimum (int value_1, int value_2, int value_3);
    
    Mat bluechannel, greenchannel, redchannel;
    vector<Mat> Channels;
    split(* BGR, Channels);
    bluechannel = Channels.at(0);
    greenchannel = Channels.at(1);
    redchannel = Channels.at(2);
    
    *DarkChannel = Mat::zeros( (*BGR).rows, (*BGR).cols, CV_8UC1);
    for( int i=0; i< (*BGR).rows ; i++)
    {
        for (int j=0; j< (*BGR).cols; j++)
        {
            (*DarkChannel).at<uchar>(i,j) = Findminimum(bluechannel.at<uchar>(i,j),
                                                     greenchannel.at<uchar>(i,j), redchannel.at<uchar>(i,j));
            
        }
    }
    //imshow("Darkchannel", *DarkChannel);
}

void qsort (Pixel * a[], int low, int high)
{
    if(low >= high)
    {
        return;
    }
    int first = low;
    int last = high;
    Pixel * key = a[first];
    while(first < last)
    {
        while((first < last)&&( (a[last]->get_gray()) <= (key->get_gray()) )) { --last; }
        a[first] = a[last];
        while((first < last)&&( (a[first]->get_gray()) >=(key->get_gray()) )) { ++first; }
        a[last] = a[first];
    }
    a[first] = key;
    qsort(a, low, first-1);
    qsort(a, first+1, high);
}

int Sum_BGR(Mat * a, int row, int col)
{
    int sum = (*a).at<Vec3b>(row,col)[0]+(*a).at<Vec3b>(row,col)[1]+(*a).at<Vec3b>(row,col)[2];
    return sum;
}

Pixel * Get_Airlight(Mat * Image)
{
    void Find_DarkChannel (Mat * RGB, Mat * DarkChannel);
    int Sum_BGR(Mat * a, int row, int col);
    
    Mat Darkchannel;
    Pixel * airlight = new Pixel();
    Pixel * airligh_copy = airlight; // to delete airlight if it change
    
    Find_DarkChannel((Image), (&Darkchannel));
    int top_size = ((*Image).rows) * ((*Image).cols) / 1000;
    Pixel * bright_list [top_size];
    
    
    for(int i=0; i<((*Image).rows);i++)
    {
        for(int j=0; j<((*Image).cols); j++)
        {
            if((i*600 + j) < top_size) // First 240 Pixels will be stored in bright_list[]
            {
                bright_list[i*600+j] = new Pixel((Darkchannel.at<uchar>(i,j)),i,j); // Constructed function Pixel(int gray, int rows, int cols)
            }
            if((i*600+j) == (top_size-1)){ qsort(bright_list, 0, top_size-1);} // Sorting for first 240 Pixels
            if((i*600+j) > (top_size-1))   // 240+ Pixels
            {
                int k = top_size-1;
                if((int)(Darkchannel.at<uchar>(i,j)) <= bright_list[k]->get_gray()) { continue; } // smaller than the last number
                
                if((int)(Darkchannel.at<uchar>(i,j)) > bright_list[k]->get_gray()) // larger than the last number
                {
                    delete bright_list[k];
                    while( (k>0) && (Darkchannel.at<uchar>(i,j) > bright_list[k-1]->get_gray()) ) // insert new pixel
                    {
                        bright_list[k] = bright_list[k-1];
                        k--;
                    }
                    bright_list[k] = new Pixel((Darkchannel.at<uchar>(i,j)),i,j);
                }
            }
        }
    }
    for (int i=0; i<top_size; i++)
    {
        if(Sum_BGR(Image, bright_list[i]->get_row(), bright_list[i]->get_col()) > airlight->get_brightness())
        { airlight = bright_list[i]; }
    }
    
    airlight->set_RGB( (*Image).at<Vec3b>(airlight->get_row(),airlight->get_col())[2],
                      (*Image).at<Vec3b>(airlight->get_row(),airlight->get_col())[1],
                      (*Image).at<Vec3b>(airlight->get_row(),airlight->get_col())[0] );
    delete airligh_copy;
    return airlight;
}

int ** get_IA(Mat * image, Pixel * Airlight)  // IA is a array[3], IA[0] stores the pointer to a array which stored the pointer of blue channel's matrix
{
    int * blue = new int[((*image).rows) * ((*image).cols)];
    int * green = new int[((*image).rows) * ((*image).cols)];
    int * red = new int[((*image).rows) * ((*image).cols)];
    int ** IA = new int * [3];

    IA[0] = blue; IA[1] = green; IA[2] = red;
    for(int i=0; i< image->rows; i++)
    {
        for(int j=0; j<image->cols; j++)
        {
            blue[i*(image->cols)+j] = (int)(*image).at<Vec3b>(i,j)[0] - (*Airlight).get_blue();
            green[i*(image->cols)+j] = (int)(*image).at<Vec3b>(i,j)[1] - (*Airlight).get_green();
            red[i*(image->cols)+j] = (int)(*image).at<Vec3b>(i,j)[2] - (*Airlight).get_red();
        }
    }
    return IA;
}

float ** Rectangle2Sphere( int ** rect , Mat * image) //IA transfer Rectangle Coordinate to Sphere, rect[0]
{                                                     //  is a pointer to a matrix correspond with IA function
    float * r = new float [((*image).rows) * ((*image).cols)];
    float * phi = new float [((*image).rows) * ((*image).cols)];
    float * theta = new float [((*image).rows) * ((*image).cols)];
    float ** sphere = new float * [3];
    
    sphere[0] = r; sphere[1] = phi; sphere[2] = theta;
    
    for(int i=0; i< image->rows; i++)
    {
        for(int j=0; j<image->cols; j++)
        {
            r[i*(image->cols)+j] = sqrt((rect[0])[i*(image->cols)+j] * (rect[0])[i*(image->cols)+j] +
                                        (rect[1])[i*(image->cols)+j] * (rect[1])[i*(image->cols)+j] +
                                        (rect[2])[i*(image->cols)+j] * (rect[2])[i*(image->cols)+j]) ;
            
            if((rect[0])[i*(image->cols)+j] == 0)
            {
                phi[i*(image->cols)+j] = (180/Pi) * atan((float)(rect[1])[i*(image->cols)+j]/0.000001);
            }
            if((rect[0])[i*(image->cols)+j] != 0)
            {
                phi[i*(image->cols)+j] = (180/Pi) * atan(((float)(rect[1])[i*(image->cols)+j])/((float)(rect[0])[i*(image->cols)+j]));
            }
            
            (r[i*(image->cols)+j] == 0) ?
            ( theta[i*(image->cols)+j] =
             acos( (rect[2])[i*(image->cols)+j] / (r[i*(image->cols)+j]+0.000001) )*180/Pi ) :
            ( theta[i*(image->cols)+j] = acos( (rect[2])[i*(image->cols)+j] / r[i*(image->cols)+j] )*180/Pi );
        }
    }
    return sphere;
}

//---------------------------------------------Kd Tree cluster

struct KdtreeNode
{
    int x,y,depth;
    float SpherePoint [3];  //contains r, phi, theta
    KdtreeNode * left, * right;
};

struct KdtreeNode * NewKdtreeNode(float r, float phi, float theta, int row, int col, int dep)
{
    struct KdtreeNode * temp = new KdtreeNode;
    temp->x = row; temp->y = col; temp->depth = dep; temp->SpherePoint[0] = r ; temp->SpherePoint[1] = phi; temp->SpherePoint[2] = theta;
    temp->left = NULL; temp->right = NULL;
    return temp;
}

struct KdtreeNode * Insert_KdtreeNode( struct KdtreeNode * root, float r, float phi, float theta, int x, int y, int dep) // insert the KdtreeNode
{
    if(root == NULL)  {return NewKdtreeNode(r, phi, theta, x, y, dep);}
    if( ((dep % 2) == 0)  )   //if( (dep % 2) == 0 )  if( ((dep % 2) == 0) && (dep < 8000) )
    {
        if(phi < root->SpherePoint[1])
            root->left = Insert_KdtreeNode( root->left, r, phi, theta, x, y, dep+1);
        else
            root->right = Insert_KdtreeNode( root->right, r, phi, theta, x, y, dep+1);
    }
    if( ((dep % 2) != 0) )      // if((dep % 2) != 0)  if( ((dep % 2) != 0) && (dep < 8000)   )
    {
        if(theta < root->SpherePoint[2])
            root->left = Insert_KdtreeNode( root->left, r, phi, theta, x, y, dep+1);
        else
            root->right = Insert_KdtreeNode( root->right, r, phi, theta, x, y, dep+1);
    }
    return root;
}

struct KdtreeNode * BuildKdtree( float ** point , Mat * image ) // the input image only used for its size
{
    struct KdtreeNode * root = NULL;
    for (int i = 0; i < (image->rows) ; i++)
    {
        for (int j = 0; j < (image->cols); j++)
        {
            root = Insert_KdtreeNode ( root, (point[0])[i*(image->cols)+j],
                                      (point[1])[i*(image->cols)+j], (point[2])[i*(image->cols)+j], i, j, 0);
        }
    }
    return root;
}

void KdtreeErgodic(KdtreeNode * root, vector<KdtreeNode *> &tree)     //???????
{
    stack<KdtreeNode *> tepps;
    KdtreeNode * p = root;
    while( ( p != NULL) || ( !tepps.empty() ) )
    {
        while( p != NULL)
        {
            tree.push_back( p );
            tepps.push(p);
            p = p->left;
        }
        if( !(tepps.empty()) )
        {
            p = tepps.top();
            tepps.pop();
            p = p->right;
        }
    }
}

vector<KdtreeNode *> * cluster_Hazeline(KdtreeNode * root, int dep) // reture a vector of node with depth (int dep) in a tree root by (KdtreeNode * root)
{
    vector<KdtreeNode *> wholetree;
    vector<KdtreeNode *> * hazeline = new vector<KdtreeNode *>;
    KdtreeErgodic(root, wholetree);

    for(vector<KdtreeNode *>::iterator tire=wholetree.begin(); tire!=wholetree.end(); tire++ )
    {
        if((*tire)->depth == dep)    { (*hazeline).push_back(*tire) ;}
    }
    return hazeline;
}

//---------------------------------------------End of Kd Tree Cluster--------------------


int * findrmax(vector<KdtreeNode*>& line) //return a array, contains the rman for each cluster
{
    int * rmax = new int[line.size()]; int i = 0;

    for(vector<KdtreeNode*>::iterator temp = line.begin(); temp!=line.end(); temp++)
    {
        vector<KdtreeNode*> treeline;
        KdtreeErgodic(*temp, treeline);
        int max = 0;
        for(vector<KdtreeNode*>::iterator temp1 = treeline.begin(); temp1 !=treeline.end(); temp1++)
        {
            if(max < (*temp1)->SpherePoint[0]) { max = (*temp1)->SpherePoint[0]; }
        }
        rmax[i] = max;
        i++;
    }
    return rmax;
}

int * rmax_show(int * rmax, vector<KdtreeNode*>& line, Mat * image) //show rmax
{
    Mat Rmax_Image = Mat::zeros((*image).rows, (*image).cols, CV_8UC1);
    int Rmaxmax = 0, i;
    for(int k = 0; k<line.size(); k++)
    {
        if(rmax[k] > Rmaxmax) Rmaxmax = rmax[k];
    }

    for(vector<KdtreeNode*>::iterator temp = line.begin(); temp!=line.end(); temp++) // clusters
    {
        vector<KdtreeNode*> treeline;
        KdtreeErgodic(*temp, treeline);
        for(vector<KdtreeNode*>::iterator temp1 = treeline.begin(); temp1 !=treeline.end(); temp1++) // in every cluster
        {
            Rmax_Image.at<uchar>((*temp1)->x,(*temp1)->y) = 255*((float)rmax[i]/(float)Rmaxmax);
        }
        i++;
    }

    imshow("rmax", Rmax_Image);
    return 0;
}

//------------------------------Concentrate Function-------------------------------------------
int * concentrate(vector<KdtreeNode*>& line, Mat * image) //input the treeline, return a array, contains the rman for each cluster
{
    Mat concentrated_image = Mat::zeros((*image).rows, (*image).cols, CV_8UC3);
    for(int i = 0; i < (*image).rows ; i++)
    {
        for(int j = 0; j < (*image).cols ; j++)
        {
            concentrated_image.at<Vec3b>(i,j)[0] = (*image).at<Vec3b>(i,j)[0];
            concentrated_image.at<Vec3b>(i,j)[1] = (*image).at<Vec3b>(i,j)[1];
            concentrated_image.at<Vec3b>(i,j)[2] = (*image).at<Vec3b>(i,j)[2];
        }
    }  // creat a new image same as input image
        
    vector<KdtreeNode*> concentration;
    
    for(vector<KdtreeNode*>::iterator temp = line.begin(); temp!=line.end(); temp++) // clusters
    {
        vector<KdtreeNode*> treeline;
        KdtreeErgodic(*temp, treeline);
        
        int max = 0;
        KdtreeNode * maxnode = *(treeline.begin());
        for(vector<KdtreeNode*>::iterator temp1 = treeline.begin(); temp1 !=treeline.end(); temp1++) // in every cluster
        {
            if(max < (*temp1)->SpherePoint[0])
            {
                max = (*temp1)->SpherePoint[0];
                maxnode = (*temp1);
            }
        }
        concentration.push_back(maxnode);
    }

    vector<KdtreeNode*>::iterator temp2 = concentration.begin();
    
    for(vector<KdtreeNode*>::iterator temp = line.begin(); temp!=line.end(); temp++) // clusters
    {
        vector<KdtreeNode*> treeline1;
        KdtreeErgodic(*temp, treeline1);
        
        for(vector<KdtreeNode*>::iterator temp1 = treeline1.begin(); temp1 !=treeline1.end(); temp1++) // in every cluster
        {
            (concentrated_image).at<Vec3b>((*temp1)->x,(*temp1)->y)[0] = (concentrated_image).at<Vec3b>((*temp2)->x,(*temp2)->y)[0];
            (concentrated_image).at<Vec3b>((*temp1)->x,(*temp1)->y)[1] = (concentrated_image).at<Vec3b>((*temp2)->x,(*temp2)->y)[1];
            (concentrated_image).at<Vec3b>((*temp1)->x,(*temp1)->y)[2] = (concentrated_image).at<Vec3b>((*temp2)->x,(*temp2)->y)[2];
        }
        temp2++;
    }
    imshow("concentration", concentrated_image);
//    imwrite("concentration.png", concentrated_image);
    return 0;
}
//----------------------------End of Concentrate Function---------------------------------------------


//-------------------------------------------------------------------------

float * get_tLB(vector <KdtreeNode*>& RootofClusters, int * rmax, Mat * image, Pixel * airlight)
{    
    float * tLB = new float[(*image).rows * (*image).cols];
    for(int i = 0; i < (*image).rows ; i++)
    {
        for(int j = 0; j < (*image).cols ; j++)
        {
            tLB[ i*((*image).cols) + j] = 1;
        }
    }
    
    for(int i = 0; i < RootofClusters.size() ; i++)
    {
        vector<KdtreeNode*> treeline; KdtreeErgodic( RootofClusters[i], treeline);
        for(int j = 0; j < treeline.size() ; j++)
        {
            float tLB1, tLB2, tLBF;
            tLB1 = ( treeline[j] )->SpherePoint[0] / rmax[i];
            tLB2 = 1 - FindminimumFloat( (float)(*image).at<Vec3b>((treeline[j])->x,(treeline[j])->y)[0] / (float)(*airlight).get_blue(),
                                        (float)(*image).at<Vec3b>((treeline[j])->x,(treeline[j])->y)[1] / (float)(*airlight).get_green(),
                                        (float)(*image).at<Vec3b>((treeline[j])->x,(treeline[j])->y)[2] / (float)(*airlight).get_red() );
            tLBF = (tLB1 > tLB2) ? tLB1 : tLB2 ;
            if(tLBF > 1) tLB[ ((treeline[j])->x) * (*image).cols + ((treeline[j])->y)] = 1;
            if(tLBF < Emin) tLB[ ((treeline[j])->x) * (*image).cols + ((treeline[j])->y)] = Emin;
            if((Emin <= tLBF)&&(tLBF <= 1)) tLB[ ((treeline[j])->x) * (*image).cols + ((treeline[j])->y)] = tLBF;
        }
    }
    return tLB;
}

Mat *  getJx (Mat * Ix, float * tLB, vector <KdtreeNode*> &treelines, Pixel * airlight)
{
    Mat * JxOutput = new Mat;
    (*JxOutput) = Mat::zeros((*Ix).rows, (*Ix).cols, CV_8UC3);
    for(int m = 0; m < (*Ix).rows ; m++)
    {
        for(int n = 0; n < (*Ix).cols; n++)
        {
            (*JxOutput).at<Vec3b>(m,n)[0] = (*Ix).at<Vec3b>(m,n)[0];
            (*JxOutput).at<Vec3b>(m,n)[1] = (*Ix).at<Vec3b>(m,n)[1];
            (*JxOutput).at<Vec3b>(m,n)[2] = (*Ix).at<Vec3b>(m,n)[2];
        }
    }
    for(int i = 0; i < treelines.size(); i++)
    {
        vector <KdtreeNode*> treeline; KdtreeErgodic( treelines[i] , treeline);
        for(int j = 0; j < treeline.size(); j++)
        {
            int color0, color1, color2;
            float tx_value = tLB[ ((treeline[j])->x) * (*Ix).cols + ((treeline[j])->y)];
            color0 = ( (*Ix).at<Vec3b>( (treeline[j])->x, (treeline[j])->y)[0] - ( 1 - (tx_value) )* (*airlight).get_blue() ) / tx_value ;
            color1 = ( (*Ix).at<Vec3b>( (treeline[j])->x, (treeline[j])->y)[1] - ( 1 - (tx_value) )* (*airlight).get_green() ) / tx_value ;
            color2 = ( (*Ix).at<Vec3b>( (treeline[j])->x, (treeline[j])->y)[2] - ( 1 - (tx_value) )* (*airlight).get_red() ) / tx_value ;
            
            (*JxOutput).at<Vec3b>((treeline[j])->x, (treeline[j])->y)[0] = (color0 > 255) ? 255 : color0;
            (*JxOutput).at<Vec3b>((treeline[j])->x, (treeline[j])->y)[1] = (color1 > 255) ? 255 : color1;
            (*JxOutput).at<Vec3b>((treeline[j])->x, (treeline[j])->y)[2] = (color2 > 255) ? 255 : color2;
            
            if( (*JxOutput).at<Vec3b>((treeline[j])->x, (treeline[j])->y)[0] < 0)
                      (*JxOutput).at<Vec3b>((treeline[j])->x, (treeline[j])->y)[0] = 0;
            if( (*JxOutput).at<Vec3b>((treeline[j])->x, (treeline[j])->y)[1] < 0)
                      (*JxOutput).at<Vec3b>((treeline[j])->x, (treeline[j])->y)[1] = 0;
            if( (*JxOutput).at<Vec3b>((treeline[j])->x, (treeline[j])->y)[2] < 0)
                      (*JxOutput).at<Vec3b>((treeline[j])->x, (treeline[j])->y)[2] = 0;

        }
    }
    return JxOutput;
}

 
/*

float variance(float * sequencelist, int size)   // Obtain the variance for a sequence of float numbers, input size is the total number of the sequence
{
    float sum = 0; float variance; float sum_power = 0 ;
    for(int i = 0; i < size ; i++) { sum = sum + sequencelist[i]; }
    float mean = sum / size;
    for(int i = 0; i< size ; i++) { sum_power = ( sequencelist[i] - mean ) * ( sequencelist[i] - mean ) ;}
    variance = sum_power / size ;
    return variance;
}

*/


void Show_Partial_Clusters(Mat * Haze, vector<KdtreeNode*> &Hazeline, int min, int max)  // show pixels belong to No.min to No.max clusters in the image
{
    Mat test = Mat::zeros((*Haze).rows, (*Haze).cols, CV_8UC3);
    for(int i = 0 ; i < test.rows ; i++)
    {
        for(int j = 0; j < test.cols; j++)
        {
            test.at<Vec3b>(i,j)[0] = 255;
            test.at<Vec3b>(i,j)[1] = 255;
            test.at<Vec3b>(i,j)[2] = 255;
        }
    }
    for(int i = min ; i < max ; i++)  // for(int i = 0 ; i < Hazeline.size() ; i++)
    {
        vector<KdtreeNode*> tepp;
        KdtreeErgodic( Hazeline[i], tepp);
        for(int j = 0; j < tepp.size(); j++)
        {
            test.at<Vec3b>( tepp[j]->x , tepp[j]->y )[0] = (*Haze).at<Vec3b>( tepp[j]->x , tepp[j]->y )[0];
            test.at<Vec3b>( tepp[j]->x , tepp[j]->y )[1] = (*Haze).at<Vec3b>( tepp[j]->x , tepp[j]->y )[1];
            test.at<Vec3b>( tepp[j]->x , tepp[j]->y )[2] = (*Haze).at<Vec3b>( tepp[j]->x , tepp[j]->y )[2];
        }
    }
    imshow("several_cluster", test);
}


int show_transmission_map(float * tLB, Mat * Image)
{
    Mat transmission_map = Mat::zeros((*Image).rows, (*Image).cols, CV_8UC1);
    for (int i = 0; i <= ((*Image).rows)-1; i++)
    {
        for (int j = 0; j <= ((*Image).cols)-1; j++)
        {
            transmission_map.at<uchar>(i,j) = 255 * tLB[ i*((*Image).cols) + j];
        }
    }
    imshow("transmission_map", transmission_map);
    
    Mat Pseudocolor_transmission_map;
    return 0;
}

int get_depth( KdtreeNode * root)
{
    int tep_depth = 0;
    vector<KdtreeNode*> all_nodes;
    KdtreeErgodic(root, all_nodes);
    for(vector<KdtreeNode*>::iterator tepp = all_nodes.begin(); tepp != all_nodes.end(); tepp++)
    {
        if ((*tepp)->depth > tep_depth ) tep_depth = (*tepp)->depth;
    }
    return tep_depth;
}

int sample_layer_number( KdtreeNode * Haze_tree, int sample_number)
{
    vector<KdtreeNode*> layers;
    KdtreeErgodic(Haze_tree, layers);
    int layer_number;

    for (int i = 0; i < get_depth(Haze_tree); i++)
    {
        int number = 0;
        for( vector<KdtreeNode*>::iterator tepp = layers.begin(); tepp != layers.end(); tepp++ )
        {
            if((*tepp)->depth == i) number++;
        }
        if(number > sample_number) { layer_number = i; break;}
    }
    return layer_number;
}

int show_rx(Mat * Haze, float ** sphereC)
{
    Mat example_Rx((*Haze).rows, (*Haze).cols, CV_8UC1);
    int max = 0;
    for (int i = 0; i < ((*Haze).rows); i++)
    {
        for (int j = 0; j < ((*Haze).cols); j++)
        {
            if (max < (sphereC[0])[i*((*Haze).cols)+j]) max = (sphereC[0])[i*((*Haze).cols)+j];
        }
    }
    for (int i = 0; i <= ((*Haze).rows)-1; i++)
    {
        for (int j = 0; j <= ((*Haze).cols)-1; j++)
        {
            example_Rx.at<uchar>(i,j) = (int)(255 * ( (sphereC[0])[i*((*Haze).cols)+j] ) / max );
        }
    }
    imshow("Rx", example_Rx);   
    return 0;
}

void show_airlight( Pixel * Haze_airlight)
{
    Mat example_airlight(256, 256, CV_8UC3);
    for (int i = 0; i <= 255; i++)
    {
        for (int j = 0; j <= 255; j++)
        {
            example_airlight.at<Vec3b>(i,j)[0] = Haze_airlight->get_blue();
            example_airlight.at<Vec3b>(i,j)[1] = Haze_airlight->get_green();
            example_airlight.at<Vec3b>(i,j)[2] = Haze_airlight->get_red();
        }
    }
    cout << Haze_airlight->get_blue() << endl;
    cout << Haze_airlight->get_green() << endl;
    cout << Haze_airlight->get_red() << endl;
    imshow("Airlight", example_airlight);
}

float * blured_tLB(float * tLB, Mat * Image)
{
    float * newtlb = new float[(*Image).rows * (*Image).cols];
    
    Mat transmission_map = Mat::zeros((*Image).rows, (*Image).cols, CV_8UC1);
    
    for (int i = 0; i <= ((*Image).rows)-1; i++)
    {
        for (int j = 0; j <= ((*Image).cols)-1; j++)
        {
            transmission_map.at<uchar>(i,j) = 255 * tLB[ i*((*Image).cols) + j];
        }
    }
    
    Mat Output;
    medianBlur(transmission_map, Output, 5);

    for (int i = 0; i <= ((*Image).rows)-1; i++)
    {
        for (int j = 0; j <= ((*Image).cols)-1; j++)
        {
            newtlb[ i*((*Image).cols) + j] = (float)( Output.at<uchar>(i,j) / (float)255);
        }
    }
    return newtlb;
}


int nonlocal_image_dehazing(Mat * Input_Image, float Airlight_Ajust, int filter, int sample_point)
{
    Mat Haze = Mat::zeros(Input_Image->rows, Input_Image->cols, CV_8UC3);
    for(int i = 0; i < Input_Image->rows; i++)
    {
        for(int j = 0; j < Input_Image->cols; j++)
        {
            Haze.at<Vec3b>(i,j)[0] = (*Input_Image).at<Vec3b>(i,j)[0];
            Haze.at<Vec3b>(i,j)[1] = (*Input_Image).at<Vec3b>(i,j)[1];
            Haze.at<Vec3b>(i,j)[2] = (*Input_Image).at<Vec3b>(i,j)[2];
        }
    }
    imshow("Input_Image", Haze);
    
    Pixel * Haze_airlight;
    Haze_airlight = Get_Airlight(&Haze);
    float xxx = Airlight_Ajust ;
    (*Haze_airlight).set_RGB( (xxx)*(*Haze_airlight).get_red(), (xxx)*(*Haze_airlight).get_green(), (xxx)*(*Haze_airlight).get_blue());
    
    show_airlight( Haze_airlight );
    //    cout << (*Haze_airlight).get_blue() << "\t" << (*Haze_airlight).get_green() << "\t" << (*Haze_airlight).get_red() << endl;
    
    int ** IA = get_IA(&Haze, Haze_airlight);    
    float ** sphereC = Rectangle2Sphere(IA, &Haze);  //    function to show rx :   show_rx( &Haze, sphereC);
    
    struct KdtreeNode * Haze_tree;
    Haze_tree = BuildKdtree(sphereC, &Haze);   // Haze_tree is the pointer to the tree's rooter
    
    vector<KdtreeNode*> tepptree;
    KdtreeErgodic(Haze_tree, tepptree);
    
    cout << "depth = " << get_depth( Haze_tree) << endl;
    
    vector <KdtreeNode*> * Hazeline;            // a vector to store each cluster's root node address
    Hazeline = cluster_Hazeline(Haze_tree, sample_layer_number(Haze_tree , sample_point));  // get Hazeline
    cout << "sample number = " << (int)((*Hazeline).size()) << endl;
    
//    Show_Partial_Clusters(&Haze, *Hazeline, 0, (int)((*Hazeline).size()) );
//    concentrate( *Hazeline, &Haze);
    
    int * rmax;     // a vector which store the
    rmax = findrmax( *Hazeline);
//    rmax_show(rmax, *Hazeline, &Haze);
    
    float * tlb;
    tlb = get_tLB( *Hazeline, rmax, &Haze, Haze_airlight);
    show_transmission_map(tlb, &Haze);
    if(filter == 1)
    {
        tlb = blured_tLB(tlb, &Haze);
    }
    Mat * finaloutput;
    finaloutput = getJx(&Haze, tlb, *Hazeline, Haze_airlight);
    imshow("output", *finaloutput);
    waitKey(-1);
    return 0;
}



//-----------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------ main function --------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------------------//


int main(int argc, char * argv[])
{
    // Haze.png, 1, 0, 1000;
    // lviv_input.jpg, 0.55, 0, 1000;
    // train_input.png, 0.55, 1, 1000;
    // farm.jpg, 0.55, 0, 1000;
    Mat Hazy_Image = imread("Haze.png");
    nonlocal_image_dehazing(&Hazy_Image, 1, 0, 1000);
    return 0;
}

//---------------------------------------------------------------------------------------------------------------------//





