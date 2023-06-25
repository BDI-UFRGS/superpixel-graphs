#include <Python.h>
#include <numpy/arrayobject.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/slic.hpp>

// graph types
enum GraphType
{
	RAG,
	KNN_SPATIAL_1,
	KNN_SPATIAL_2,
	KNN_SPATIAL_4,
	KNN_SPATIAL_8,
	KNN_SPATIAL_16,
	KNN_FEATURE_1,
	KNN_FEATURE_2,
	KNN_FEATURE_4,
	KNN_FEATURE_8,
	KNN_FEATURE_16
};

enum DistanceMeasure
{
    SPATIAL,
    FEATURE
};

// segmentation methods
enum SegmentationMethod
{
	ADAPTATIVE_SLIC,
	NORMAL_SLIC,
    GRID
};

// grayscale features
#define GRAY_AVG_COLOR 0
#define GRAY_STD_DEV_COLOR 1
#define GRAY_CENTROID_I 2
#define GRAY_CENTROID_J 3
#define GRAY_STD_DEV_CENTROID_I 4
#define GRAY_STD_DEV_CENTROID_J 5
#define GRAY_NUM_PIXELS 6

#define FEATURES_GRAYSCALE 7

// color features
#define COLOR_AVG_COLOR_R 0
#define COLOR_AVG_COLOR_G 1
#define COLOR_AVG_COLOR_B 2
#define COLOR_STD_DEV_COLOR_R 3
#define COLOR_STD_DEV_COLOR_G 4
#define COLOR_STD_DEV_COLOR_B 5
#define COLOR_CENTROID_I 6
#define COLOR_CENTROID_J 7
#define COLOR_STD_DEV_CENTROID_I 8
#define COLOR_STD_DEV_CENTROID_J 9
#define COLOR_NUM_PIXELS 10
#define COLOR_AVG_COLOR_H 11
#define COLOR_AVG_COLOR_S 12
#define COLOR_AVG_COLOR_V 13
#define COLOR_STD_DEV_COLOR_H 14
#define COLOR_STD_DEV_COLOR_S 15
#define COLOR_STD_DEV_COLOR_V 16

#define FEATURES_COLOR 17


PyArrayObject *get_edge_index(cv::Mat s, cv::Mat features, GraphType graph_type, int img_width, int img_height);
std::vector<std::pair<int, int>> RAG_adj(cv::Mat s);
std::vector<std::pair<int, int>> KNN_adj(cv::Mat s, cv::Mat features, int k, DistanceMeasure distance_measure, int img_width, int img_height);
float compute_distance(int u, int v, cv::Mat features, DistanceMeasure distance_measure, int img_width, int img_height);
float feature_distance(int u, int v, cv::Mat features, int img_width, int img_height);
float spatial_distance(int u, int v, cv::Mat features);

cv::Mat grayscale_features(cv::Mat s, int n, cv::Mat img);
cv::Mat color_features(cv::Mat s, int n, cv::Mat img);

cv::Mat from_numpy(PyArrayObject *a);
PyArrayObject *to_numpy_int32(cv::Mat a);
PyArrayObject *to_numpy_float64(cv::Mat a);
