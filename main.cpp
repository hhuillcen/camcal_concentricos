#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<time.h>

using namespace cv;
using namespace std;

#define IMAGE_WIDTH  640
#define IMAGE_HEIGHT 480
#define S (IMAGE_WIDTH/8)
#define T (0.15f)
IplImage* cvFrame;
IplImage* binImg;
int key;


Mat src, src_gray;
int thresh = 140;
int max_thresh = 255;
int frames_correctos=0;
vector<RotatedRect> sepEllipse(22);
vector<RotatedRect> resEllipse(22);


Mat temp = Mat::zeros( 480, 640, CV_8UC3 );
Mat temp1 = Mat::zeros( 480, 640, CV_8UC3 );
vector<vector<Point2f>> coord2D;
float dist_x = 44.3;
float dist_y = 44.3;
vector<Point3f> puntos3D;
vector<vector<Point3f>> coord3D;
int capturas=0;

Mat intrinsic = Mat(3, 3, CV_32FC1);
Mat distCoeffs;
vector<Mat> rvecs;
vector<Mat> tvecs;
Size patternsize(5,4);

int total_puntos_computados=0;
double distancia=0;
double total_distancias=0;

int nro_imagenes=45;
int cont_imag=0;
int puntos_patron=20;

double distancia_punto_recta(Point2f p1,Point2f p2, Point2f clave)
{
		double m=(p2.y-p1.y)/(p2.x-p1.x);
		double b=p1.y-(m*p1.x);
		double dist=(abs(m*clave.x-clave.y+b)) / (sqrt((m*m)+1));
		return dist;
}


static double computeReprojectionErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs
         )
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    //perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
		//cout<<" \n objectpoints "<<objectPoints[i]<<"\n"<<endl;
		//cout<<" \n imagepoints2 "<<imagePoints2[i]<<"\n"<<endl;
		//
				for (int x=0;x<=imagePoints2.size();x++)   // muestra los puntos detectados en cada frame
				{
				ostringstream convert; 
				convert<<x;
				circle(temp1, imagePoints2[x],1, Scalar(0,0,255),CV_FILLED, 8,0);
				putText(temp1, convert.str(), imagePoints2[x], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
				}
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);

		//cout<<" \n imagepoint de : "<<i<<"\n"<<imagePoints2[i]<<"\n"<<endl;
		namedWindow( "Temporal1", CV_WINDOW_AUTOSIZE );
		imshow( "Temporal1", temp1 );
		//waitKey();
        int n = (int)objectPoints[i].size();
        totalErr += err*err;
        totalPoints += n;
    }

    return sqrt(totalErr/totalPoints);
}


static double computeColiErrors(
        const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs
         )
{
    vector<Point2f> imagePoints2;
	int i;
    

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		
		//colinearidad entre 4 y 0
		for (int x=1; x<=3;x++)
		{
			distancia= distancia_punto_recta(imagePoints2[4],imagePoints2[0],imagePoints2[x]);
			total_distancias=total_distancias+distancia;
			total_puntos_computados=total_puntos_computados+1;
		}

		//colinearidad entre 9 y 5
		for (int x=6; x<=8;x++)
		{
			distancia= distancia_punto_recta(imagePoints2[9],imagePoints2[5],imagePoints2[x]);
			total_distancias=total_distancias+distancia;
			total_puntos_computados=total_puntos_computados+1;
		}

		//colinearidad entre 14 y 10
		for (int x=11; x<=13;x++)
		{
			distancia= distancia_punto_recta(imagePoints2[14],imagePoints2[10],imagePoints2[x]);
			total_distancias=total_distancias+distancia;
			total_puntos_computados=total_puntos_computados+1;
		}

		//colinearidad entre 19 y 15
		for (int x=16; x<=18;x++)
		{
			distancia= distancia_punto_recta(imagePoints2[19],imagePoints2[15],imagePoints2[x]);
			total_distancias=total_distancias+distancia;
			total_puntos_computados=total_puntos_computados+1;
		}
    }

    return total_distancias/total_puntos_computados;
}


void adaptiveThreshold(unsigned char* input, unsigned char* bin)
{
	unsigned long* integralImg = 0;
	int i, j;
	long sum=0;
	int count=0;
	int index;
	int x1, y1, x2, y2;
	int s2 = S/2;

	// create the integral image
	integralImg = (unsigned long*)malloc(IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(unsigned long*));  //imagen integral en blanco

	for (i=0; i<IMAGE_WIDTH; i++)
	{
		// reset this column sum
		sum = 0;

		for (j=0; j<IMAGE_HEIGHT; j++)
		{
			index = j*IMAGE_WIDTH+i;

			sum += input[index];
			if (i==0)
				integralImg[index] = sum;
			else
				integralImg[index] = integralImg[index-1] + sum;
		}
	}

	// perform thresholding
	for (i=0; i<IMAGE_WIDTH; i++)
	{
		for (j=0; j<IMAGE_HEIGHT; j++)
		{
			index = j*IMAGE_WIDTH+i;

			// set the SxS region
			x1=i-s2; x2=i+s2;
			y1=j-s2; y2=j+s2;

			// check the border
			if (x1 < 0) x1 = 0;
			if (x2 >= IMAGE_WIDTH) x2 = IMAGE_WIDTH-1;
			if (y1 < 0) y1 = 0;
			if (y2 >= IMAGE_HEIGHT) y2 = IMAGE_HEIGHT-1;
			
			count = (x2-x1)*(y2-y1);

			// I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
			sum = integralImg[y2*IMAGE_WIDTH+x2] -
				  integralImg[y1*IMAGE_WIDTH+x2] -
				  integralImg[y2*IMAGE_WIDTH+x1] +
				  integralImg[y1*IMAGE_WIDTH+x1];

			if ((long)(input[index]*count) < (long)(sum*(1.0-T)))
				bin[index] = 0;
			else
				bin[index] = 255;
		}
	}

	free (integralImg);
}



/** @function reconocer_elipses */
void reconocer_elipses(int, void*) 
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  vector<Point2f> puntos;

    

  /// Detectar contornos usando Threshold, osea segmentado
  //threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );

  //Declaración de los parámetros(imagenes iplimage) para la funcion adaptativeThreshold

  binImg = cvCreateImage(cvSize(IMAGE_WIDTH, IMAGE_HEIGHT), 8, 1);  //en blanco
  cvFrame = (IplImage*)(&IplImage(src_gray)); // convierte la matriz imagen src_gray a tipo IplImage

  adaptiveThreshold((unsigned char*)cvFrame->imageData, (unsigned char*)binImg->imageData);

  threshold_output = cvarrToMat(binImg);

  //imshow( "Imagen segmentada", threshold_output );

  /// Encontrar contornos
  findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );

  /// Encontrar rectangulos rotados y elipses por cada contorno
  vector<RotatedRect> minEllipse(contours.size() );


  for( int i = 0; i < contours.size(); i++ )
     {
       if( (contours[i].size() > 5) && (contours[i].size() < 100) )
         {
			 minEllipse[i] = fitEllipse( Mat(contours[i]) ); 
	     }
     }

  
  /// Crear imagen para mostrar elipses

  Mat drawing(480, 640, CV_8UC3, Scalar(255,255,255));
  int cont=1;
  float tempx=0;
  float tempy=0;
  int estado=1;
  for( int i = 1; i< contours.size()-1; i++ )
     {
       Scalar color = Scalar( 255, 255,255 );
       
       float centrox;
	   float centroy;
	   float dif_centros_ejex_sig = abs(minEllipse[i].center.x - minEllipse[i+1].center.x);
	   float dif_centros_ejey_sig = abs(minEllipse[i].center.y - minEllipse[i+1].center.y);
	   float dif_centros_ejex_ant = abs(minEllipse[i].center.x - minEllipse[i-1].center.x);
	   float dif_centros_ejey_ant = abs(minEllipse[i].center.y - minEllipse[i-1].center.y);
	   float proporcion=minEllipse[i].size.width/minEllipse[i].size.height;

	   if (((dif_centros_ejex_sig<=1 && dif_centros_ejey_sig<=1) || (dif_centros_ejex_ant<=1 && dif_centros_ejey_ant<=1))&&(proporcion<=1.6&&proporcion>=0.5))
	   {
		  if (centrox=minEllipse[i].center.x!=0)
		   {
		   centrox=minEllipse[i].center.x;
		   centroy=minEllipse[i].center.y;
		   //ellipse( drawing, minEllipse[i], color, 1,255 );

		   
		   
		   //cout<<"Centro (X,Y) anillo "<<cont<< " :  ( " <<centrox<<" , "<<centroy<<" ) "  << endl;

		   if (estado==-1)
			   {
				//cout<<"diferencia de centros X , Y : "<<centrox-tempx<<", "<<centroy-tempy<<endl;		
				// line(drawing, Point(20,20),Point(200,200),color);
				float prom_x=(centrox+tempx)/2;
				float prom_y=(centroy+tempy)/2;
				sepEllipse[cont]=minEllipse[i];
				sepEllipse[cont].center.x=prom_x;
				sepEllipse[cont].center.y=prom_y;
				//ellipse( drawing, sepEllipse[cont], color, 1,255 );
				cont=cont+1;
			   }
		   tempx=centrox;
		   tempy=centroy;
		   estado=estado*-1;

		   }
	   }

     }



	for(int i=1;i<=20;i++)
	{
		if (sepEllipse[i].center.x!=0){
		//line(drawing, sepEllipse[i].center,sepEllipse[i+1].center,Scalar( 0, 255,255 ));
			//circle(img, Point(50,50),50, Scalar(255,255,255),CV_FILLED, 8,0);
			circle(drawing, sepEllipse[i].center, 10, Scalar( 0,0, 0 ),CV_FILLED, 8,0);}
	}

	//parámetros para findCirclesGrid

	SimpleBlobDetector::Params params;
	params.thresholdStep = 10;
	params.minThreshold = 50;
	params.maxThreshold = 255;
	params.minRepeatability = 1;
	params.maxArea = 10000; // 100 * 100
	params.minArea = 100; // 10 * 10
	Ptr<FeatureDetector> blobDetector = new SimpleBlobDetector(params);

	//Size patternsize(5,4);
	bool deteccion;
			
	//Mat image = imread("d://casa//circles.jpg");
	//deteccion = findCirclesGrid(image, patternsize, puntos, CALIB_CB_ASYMMETRIC_GRID); //| CALIB_CB_CLUSTERING, blobDetector);
	deteccion = findCirclesGrid(drawing, patternsize, puntos, CALIB_CB_SYMMETRIC_GRID | CALIB_CB_CLUSTERING, blobDetector);





	if (deteccion == false )
	{
		cout << "la deteccion fallo!" << endl;
		//return 0;
	}
	else
	{
		  //calibracion de camara
		if (puntos.size()==puntos_patron)
		{
		cout << "Exitoso!" << endl;
		drawChessboardCorners(src,patternsize,puntos,true);   //dibuja los centros y las lineas de colores


		  //if(waitKey(1) == 99)  // detectar pulsacion de la tecla "c", para capturar las coordenadas 2D
			 // {

				for (int x=0;x<=puntos.size();x++)   // muestra los puntos detectados en cada frame
				{
				//ostringstream convert; 
				//convert<<x;
				circle(temp, puntos[x],1, Scalar(0,0,255),CV_FILLED, 8,0);
				//putText(temp, convert.str(), puntos[x], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
				}
				  namedWindow( "Temporal", CV_WINDOW_AUTOSIZE );
				  imshow( "Temporal", temp );

				  //grabar las coordenadas 2D 

				  coord2D.push_back(puntos);
				  capturas=capturas+1;
				  //waitKey();
		//	 }

		 // namedWindow( "Reconocimiento", CV_WINDOW_AUTOSIZE );
		  imshow( "Original", src );
		  cont_imag=cont_imag+1;
		  cout<<"imagen valida nro: "<<cont_imag<<endl;
		  waitKey(30);
		  }


	}
}








/** @function main */
int main( int argc, char** argv )
{


	for(int i=0; i<patternsize.height;i++){
		for(int j=0; j<patternsize.width;j++) {
			puntos3D.push_back(Point3f(j*dist_x, i*dist_y, 0)); //adiciona un elemento al final
		}
	}
	//intrinsic.ptr<float>(0)[0] = 1;
 //   intrinsic.ptr<float>(1)[1] = 1;

	namedWindow( "OpenCV Video", CV_WINDOW_AUTOSIZE);

	// cargar el archivo de video especificado
	VideoCapture vc("d://casa//Rings.mp4");


	// verificar si se ha podio cargar el video
	if(!vc.isOpened()) return -1;
	// obtener los cuadros por segundo
	//double fps = vc.get(CV_CAP_PROP_FPS);

	//cout<<"velocidad es: "<<fps<<endl;

	//int c=0;
	//int a=1;
	//int pivote=166; //para 50 imagenes

	int frnb ( vc.get ( CV_CAP_PROP_FRAME_COUNT ) );
    std::cout << "Numero de Frames = " << frnb << endl;

	
	//waitKey();

	int leer_frame;

	while (true)
	{

		//Mat frame;
		//vc >> frame;
		//c=c+1;

		//srand(time(NULL));
		leer_frame=1+rand()%(frnb-1);
		//waitKey();
		Mat frame;
		vc.set ( CV_CAP_PROP_POS_FRAMES , leer_frame );
		vc.read(frame);

		imshow( "Original", frame);

		//if ((c==1)||(c==240)||(c==450)||(c==520)||(c==750)||(c==890)||(c==914)||(c==980)
		//	||(c==1001)||(c==1240)||(c==1480)||(c==1520)||(c==1600)||(c==1890)||(c==1914)||(c==2000)
		//	||(c==2001)||(c==2280)||(c==2480)||(c==2520)||(c==2750)||(c==2890)||(c==2914)||(c==3000)||(c==3080))
		//if (c==pivote*a)


		if(cont_imag<nro_imagenes)
		{
			cout<<endl<<"PROCESAMIENTO DE FRAME NRO :  "<< leer_frame <<endl<<endl;

			src = frame.clone();

			/// Convertir imagen a escala de grises y luego suavizar
			cvtColor( src, src_gray, CV_BGR2GRAY );
			blur( src_gray, src_gray, Size(3,3) );


			//imshow( "source", src_gray );

			//crear la barra de segmentacion
			//createTrackbar( " Umbral de segmemtación:", "source", &thresh, max_thresh, reconocer_elipses );
			reconocer_elipses(0,0);
			//a=a+1;
			
			//waitKey();
		}

		if(cont_imag==nro_imagenes)
		{
			//calibrar la camara
			cout<<endl<<"EMPEZANDO CALIBRACION CON  :  "<< cont_imag <<"  imagenes correctamente detectadas"<<endl;
			  //destroyWindow("Reconocimiento");
			  //destroyWindow("Temporal");
			  coord3D.resize(cont_imag,puntos3D);
			  //Mat distCoeffs;
			  //vector<Mat> rvecs, tvecs;
				  //Intrínsecos:
				//Mat cameraMatrix;//(3,3,CV_64FC1);
				//Mat distCoeffs;//(8,1,CV_64FC1);
				//Extrínsecos:
				
			 
			  intrinsic = initCameraMatrix2D(coord3D, coord2D, src.size());
				//double rms = cv::calibrateCamera(coord3D, coord2D, Size(5,4), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_RATIONAL_MODEL);
			  calibrateCamera(coord3D, coord2D,src.size(), intrinsic, distCoeffs, rvecs, tvecs);
				
				cout<<"Coeficientes de Distorsión \n"<<distCoeffs<<endl;

				cout<<"Matriz de la camara \n"<<intrinsic<<endl;
				waitKey();

				//Calcular el error
				

				//
				double totalAvgErr = computeReprojectionErrors(coord3D, coord2D, rvecs, tvecs, intrinsic, distCoeffs);

				cout << "\n Error estandar de prediccion promedio: " << totalAvgErr<< endl;
				waitKey();

				double totalColiErr = computeColiErrors(coord3D, coord2D, rvecs, tvecs, intrinsic, distCoeffs);

				cout << "\n Promedio de colinearidad de los puntos " << totalColiErr<<endl;
				waitKey();

				cout << "\n Pulse una tecla para empezar con los resultados.." << endl;
				waitKey();

				cont_imag=cont_imag+1;
				//return 0;
				//vector<Point2f> projectedPoints;
				//projectPoints(coord3D[2], rvecs[2], tvecs[2], intrinsic, distCoeffs, projectedPoints);
				//for (int i = 0; i < projectedPoints.size(); ++i)
				//{
				//	cout << "Image point: " << coord2D[i] << " Projected to " << projectedPoints[i] << endl;
				//}			
				//waitKey();
				
				//VideoCapture cap(0);  //apertura la webcam
				//if(!cap.isOpened())
				//	return -1;

				//VideoCapture vc1("d://casa//Rings.mp4");


				// verificar si se ha podio cargar el video
				//if(!vc1.isOpened()) return -1;

				//for(;;)
				//{
				//	Mat frame1;
				//	vc1>> frame1;
				//	Mat imageUndistorted;
				//	undistort(frame1, imageUndistorted, intrinsic, distCoeffs);
				//	imshow( "Original", frame1);
				//	imshow("Calibrado", imageUndistorted);
				//	waitKey(30);

				//}



				
		}


		if (cont_imag>nro_imagenes)
		{
				VideoCapture vc1("d://casa//Rings.mp4");
				if(!vc1.isOpened()) return -1;
				for(;;)
				{
					Mat frame1;
					vc1>> frame1;
					Mat imageUndistorted;
					undistort(frame1, imageUndistorted, intrinsic, distCoeffs);
					imshow( "Original", frame1);
					imshow("Calibrado", imageUndistorted);
					waitKey(30);

				}

		}


	  
	}
	
}






