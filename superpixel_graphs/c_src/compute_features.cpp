#include "compute_features.h"

#include <iostream>
#include <cmath>
#include <unordered_set>
#include <set>
#include <utility>

cv::Mat grayscale_features(cv::Mat s, int n, cv::Mat img)
{
    cv::Mat s1 = cv::Mat::zeros(n, 1, CV_64F),
            s2 = cv::Mat::zeros(n, 1, CV_64F),
            posi1 = cv::Mat::zeros(n, 1, CV_64F),
            posj1 = cv::Mat::zeros(n, 1, CV_64F),
            posi2 = cv::Mat::zeros(n, 1, CV_64F),
            posj2 = cv::Mat::zeros(n, 1, CV_64F),
            num_pixels = cv::Mat::zeros(n, 1, CV_64F);
    int node;
    float color;
    for (int i=0; i<img.rows; i++)
        for (int j=0; j<img.cols; j++)
        {
            node = s.at<int32_t>(i, j);
            color = img.at<float>(i,j)/255;
            s1.at<double>(node,0) += color;
            s2.at<double>(node,0) += pow(color, 2);
            posi1.at<double>(node,0) += i;
            posj1.at<double>(node,0) += j;
            posi2.at<double>(node,0) += pow(i, 2);
            posj2.at<double>(node,0) += pow(j, 2);
            num_pixels.at<double>(node,0) += 1;
        }
    cv::Mat features(n, FEATURES_GRAYSCALE, CV_64F);
    // color features
    cv::divide(s1, num_pixels, s1);
    cv::divide(s2, num_pixels, s2);
    cv::Mat avg_color = s1;
    avg_color = avg_color;
    cv::Mat std_dev_color = cv::abs(s2 - s1.mul(s1));
    cv::sqrt(std_dev_color, std_dev_color);
    std_dev_color = std_dev_color;
    avg_color.copyTo(features.col(GRAY_AVG_COLOR));
    std_dev_color.copyTo(features.col(GRAY_STD_DEV_COLOR));
    // positional features
    cv::divide(posi1, num_pixels, posi1);
    cv::divide(posj1, num_pixels, posj1);
    cv::divide(posi2, num_pixels, posi2);
    cv::divide(posj2, num_pixels, posj2);
    cv::Mat centroid_i = posi1;
    cv::Mat centroid_j = posj1;
    cv::Mat std_dev_centroid_i = cv::abs(posi2 - posi1.mul(posi1));
    cv::sqrt(std_dev_centroid_i, std_dev_centroid_i);
    cv::Mat std_dev_centroid_j = cv::abs(posj2 - posj1.mul(posj1));
    cv::sqrt(std_dev_centroid_j, std_dev_centroid_j);
    centroid_i.copyTo(features.col(GRAY_CENTROID_I));
    centroid_j.copyTo(features.col(GRAY_CENTROID_J));
    std_dev_centroid_i.copyTo(features.col(GRAY_STD_DEV_CENTROID_I));
    std_dev_centroid_j.copyTo(features.col(GRAY_STD_DEV_CENTROID_J));

    num_pixels.copyTo(features.col(GRAY_NUM_PIXELS));

    return features;
}

cv::Mat color_features(cv::Mat s, int n, cv::Mat img)
{
    cv::Mat img_hsv;
    cv::cvtColor(img, img_hsv, cv::COLOR_RGB2HSV);

    std::vector<cv::Mat> rgb(3), hsv(3);
    cv::split(img, rgb);
    cv::split(img_hsv, hsv);
    cv::Mat sr1 = cv::Mat::zeros(n, 1, CV_64F),
            sg1 = cv::Mat::zeros(n, 1, CV_64F),
            sb1 = cv::Mat::zeros(n, 1, CV_64F),
            sr2 = cv::Mat::zeros(n, 1, CV_64F),
            sg2 = cv::Mat::zeros(n, 1, CV_64F),
            sb2 = cv::Mat::zeros(n, 1, CV_64F),
            posi1 = cv::Mat::zeros(n, 1, CV_64F),
            posj1 = cv::Mat::zeros(n, 1, CV_64F),
            posi2 = cv::Mat::zeros(n, 1, CV_64F),
            posj2 = cv::Mat::zeros(n, 1, CV_64F),
            num_pixels = cv::Mat::zeros(n, 1, CV_64F),
            sh1 = cv::Mat::zeros(n, 1, CV_64F),
            ss1 = cv::Mat::zeros(n, 1, CV_64F),
            sv1 = cv::Mat::zeros(n, 1, CV_64F),
            sh2 = cv::Mat::zeros(n, 1, CV_64F),
            ss2 = cv::Mat::zeros(n, 1, CV_64F),
            sv2 = cv::Mat::zeros(n, 1, CV_64F);
    int node;
    float colorR, colorG, colorB, colorH, colorS, colorV;
    for (int i=0; i<img.rows; i++)
        for (int j=0; j<img.cols; j++)
        {
            node = s.at<int32_t>(i, j);
            colorR = rgb[0].at<float>(i, j)/255;
            colorG = rgb[1].at<float>(i, j)/255;
            colorB = rgb[2].at<float>(i, j)/255;
            colorH = hsv[0].at<float>(i, j)/255;
            colorS = hsv[1].at<float>(i, j)/255;
            colorV = hsv[2].at<float>(i, j)/255;
            sr1.at<double>(node,0) += colorR;
            sr2.at<double>(node,0) += pow(colorR, 2);
            sg1.at<double>(node,0) += colorG;
            sg2.at<double>(node,0) += pow(colorG, 2);
            sb1.at<double>(node,0) += colorB;
            sb2.at<double>(node,0) += pow(colorB, 2);
            sh1.at<double>(node,0) += colorH;
            sh2.at<double>(node,0) += pow(colorH, 2);
            ss1.at<double>(node,0) += colorS;
            ss2.at<double>(node,0) += pow(colorS, 2);
            sv1.at<double>(node,0) += colorV;
            sv2.at<double>(node,0) += pow(colorV, 2);
            posi1.at<double>(node,0) += i;
            posj1.at<double>(node,0) += j;
            posi2.at<double>(node,0) += pow(i, 2);
            posj2.at<double>(node,0) += pow(j, 2);
            num_pixels.at<double>(node,0) += 1;
        }
    cv::Mat features(n, FEATURES_COLOR, CV_64F);
    // color features
    cv::divide(sr1, num_pixels, sr1);
    cv::divide(sr2, num_pixels, sr2);
    cv::divide(sg1, num_pixels, sg1);
    cv::divide(sg2, num_pixels, sg2);
    cv::divide(sb1, num_pixels, sb1);
    cv::divide(sb2, num_pixels, sb2);
    cv::divide(sh1, num_pixels, sh1);
    cv::divide(sh2, num_pixels, sh2);
    cv::divide(ss1, num_pixels, ss1);
    cv::divide(ss2, num_pixels, ss2);
    cv::divide(sv1, num_pixels, sv1);
    cv::divide(sv2, num_pixels, sv2);
    
    cv::Mat avg_color_r = sr1;
    cv::Mat std_dev_color_r = cv::abs(sr2 - sr1.mul(sr1));
    cv::sqrt(std_dev_color_r, std_dev_color_r);
    avg_color_r.copyTo(features.col(COLOR_AVG_COLOR_R));
    std_dev_color_r.copyTo(features.col(COLOR_STD_DEV_COLOR_R));

    cv::Mat avg_color_g = sg1;
    cv::Mat std_dev_color_g = cv::abs(sg2 - sg1.mul(sg1));
    cv::sqrt(std_dev_color_g, std_dev_color_g);
    avg_color_g.copyTo(features.col(COLOR_AVG_COLOR_G));
    std_dev_color_g.copyTo(features.col(COLOR_STD_DEV_COLOR_G));
    
    cv::Mat avg_color_b = sb1;
    cv::Mat std_dev_color_b = cv::abs(sb2 - sb1.mul(sb1));
    cv::sqrt(std_dev_color_b, std_dev_color_b);
    avg_color_b.copyTo(features.col(COLOR_AVG_COLOR_B));
    std_dev_color_b.copyTo(features.col(COLOR_STD_DEV_COLOR_B));
    
    cv::Mat avg_color_h = sh1;
    cv::Mat std_dev_color_h = cv::abs(sh2 - sh1.mul(sh1));
    cv::sqrt(std_dev_color_h, std_dev_color_h);
    avg_color_h.copyTo(features.col(COLOR_AVG_COLOR_H));
    std_dev_color_h.copyTo(features.col(COLOR_STD_DEV_COLOR_H));
    
    cv::Mat avg_color_s = ss1;
    cv::Mat std_dev_color_s = cv::abs(ss2 - ss1.mul(ss1));
    cv::sqrt(std_dev_color_s, std_dev_color_s);
    avg_color_s.copyTo(features.col(COLOR_AVG_COLOR_S));
    std_dev_color_s.copyTo(features.col(COLOR_STD_DEV_COLOR_S));
    
    cv::Mat avg_color_v = sv1;
    cv::Mat std_dev_color_v = cv::abs(sv2 - sv1.mul(sv1));
    cv::sqrt(std_dev_color_v, std_dev_color_v);
    avg_color_v.copyTo(features.col(COLOR_AVG_COLOR_V));
    std_dev_color_v.copyTo(features.col(COLOR_STD_DEV_COLOR_V));

    // positional features
    cv::divide(posi1, num_pixels, posi1);
    cv::divide(posj1, num_pixels, posj1);
    cv::divide(posi2, num_pixels, posi2);
    cv::divide(posj2, num_pixels, posj2);
    cv::Mat centroid_i = posi1;
    cv::Mat centroid_j = posj1;
    cv::Mat std_dev_centroid_i = cv::abs(posi2 - posi1.mul(posi1));
    cv::sqrt(std_dev_centroid_i, std_dev_centroid_i);
    cv::Mat std_dev_centroid_j = cv::abs(posj2 - posj1.mul(posj1));
    cv::sqrt(std_dev_centroid_j, std_dev_centroid_j);
    centroid_i.copyTo(features.col(COLOR_CENTROID_I));
    centroid_j.copyTo(features.col(COLOR_CENTROID_J));
    std_dev_centroid_i.copyTo(features.col(COLOR_STD_DEV_CENTROID_I));
    std_dev_centroid_j.copyTo(features.col(COLOR_STD_DEV_CENTROID_J));

    num_pixels.copyTo(features.col(COLOR_NUM_PIXELS));

    return features;

}


std::vector<std::pair<int, int>> RAG_adj(cv::Mat s)
{
    std::set<std::pair<int, int>> adj;
    int current, other;
    for (int i=0; i<s.rows; i++)
        for (int j=0; j<s.cols; j++)
        {
            current = s.at<int32_t>(i,j);
            if(i-1 >= 0 && current != s.at<int32_t>(i-1, j))
            {
                other = s.at<int32_t>(i-1, j);
                std::pair<int, int> edge = std::make_pair(current, other);
                adj.insert(edge);
            }
            if(i+1 < s.rows && current != s.at<int32_t>(i+1, j))
            {
                other = s.at<int32_t>(i+1, j);
                std::pair<int, int> edge = std::make_pair(current, other);
                adj.insert(edge);
            }
            if(j-1 >= 0 && current != s.at<int32_t>(i, j-1))
            {
                other = s.at<int32_t>(i, j-1);
                std::pair<int, int> edge = std::make_pair(current, other);
                adj.insert(edge);
            }
            if(j+1 < s.cols && current != s.at<int32_t>(i, j+1))
            {
                other = s.at<int32_t>(i, j+1);
                std::pair<int, int> edge = std::make_pair(current, other);
                adj.insert(edge);
            }
        }
    std::vector<std::pair<int, int>> adj_vec(adj.begin(), adj.end());
    return adj_vec;
}

float compute_distance(int u, int v, cv::Mat features, DistanceMeasure distance_measure, int img_width, int img_height)
{
    if(u == v)
        return 0.0;

    switch (distance_measure)
    {
    case SPATIAL:
        return spatial_distance(u, v, features);
    case FEATURE:
        return feature_distance(u, v, features, img_width, img_height);
    }
    return 0.0;
}

float spatial_distance(int u, int v, cv::Mat features)
{
    float d;
    int i,j;
    if(features.cols == FEATURES_GRAYSCALE)
    {
        i = GRAY_CENTROID_I;
        j = GRAY_CENTROID_J;
    }
    else 
    {
        i = COLOR_CENTROID_I;
        j = COLOR_CENTROID_J;
    }
    d = sqrt(  pow(features.at<double>(u, i) - features.at<double>(v, i), 2) 
             + pow(features.at<double>(u, j) - features.at<double>(v, j), 2));
    return d;
}

float feature_distance(int u, int v, cv::Mat features, int img_width, int img_height)
{
    // using normalized avg. color and centroid distance
    float d;
    if(features.cols == FEATURES_GRAYSCALE)
    {
        int i, j, l;
        i = GRAY_CENTROID_I;
        j = GRAY_CENTROID_J;
        l = GRAY_AVG_COLOR;
        d =   sqrt((  pow((features.at<double>(u, i) - features.at<double>(v, i))/img_height, 2) 
                    + pow((features.at<double>(u, j) - features.at<double>(v, j))/img_width, 2))/2.0
                    + pow(features.at<double>(u, l) - features.at<double>(v, l), 2));
    }
    else
    {
        int i, j, r, g, b;
        i = COLOR_CENTROID_I;
        j = COLOR_CENTROID_J;
        r = COLOR_AVG_COLOR_R;
        g = COLOR_AVG_COLOR_G;
        b = COLOR_AVG_COLOR_B;
        d =   sqrt(  (pow((features.at<double>(u, i) - features.at<double>(v, i))/img_height, 2) 
                   +  pow((features.at<double>(u, j) - features.at<double>(v, j))/img_width, 2))/2.0
                   + (pow(features.at<double>(u, r) - features.at<double>(v, r), 2)
                   +  pow(features.at<double>(u, g) - features.at<double>(v, g), 2)
                   +  pow(features.at<double>(u, b) - features.at<double>(v, b), 2))/3.0);
    }
    return d;
}

std::vector<std::pair<int, int>> KNN_adj(cv::Mat s, cv::Mat features, int k, DistanceMeasure distance_measure, int img_width, int img_height)
{
    std::vector<std::pair<int, int>> adj;
    int n = features.rows;
    std::vector<std::pair<float, int>> distances(n);

    for(int u = 0; u<n; u++)
    {
        // initialize distances vector
        for(int v=0; v<n; v++)
        {
            distances[v].first = compute_distance(u, v, features, distance_measure, img_width, img_height);
            distances[v].second = v;
        }
        std::sort(distances.begin(), distances.end());

        // get k nearest (not including itself)
        int ks = 0, i = 0;
        while(ks < k)
        {
            if(u != distances[i].second)
            {
                std::pair<int, int> edge0, edge1;
                edge0.first = u;
                edge0.second = distances[i].second;
                edge1.first = distances[i].second;
                edge1.second = u;
                adj.push_back(edge0);
                adj.push_back(edge1);
                ks++;
            }
            i++;
        }
    }
    return adj;  
}

PyArrayObject *get_edge_index(cv::Mat s, cv::Mat features, GraphType graph_type, int img_width, int img_height)
{
    std::vector<std::pair<int, int>> adj;
    switch (graph_type)
    {
	case RAG:
        adj = RAG_adj(s);
		break;
	case KNN_SPATIAL_1:
		adj = KNN_adj(s, features, 1, SPATIAL, img_width, img_height);
		break;
	case KNN_SPATIAL_2:
		adj = KNN_adj(s, features, 2, SPATIAL, img_width, img_height);
		break;
	case KNN_SPATIAL_4:
		adj = KNN_adj(s, features, 4, SPATIAL, img_width, img_height);
		break;
	case KNN_SPATIAL_8:
		adj = KNN_adj(s, features, 8, SPATIAL, img_width, img_height);
		break;
	case KNN_SPATIAL_16:
		adj = KNN_adj(s, features, 16, SPATIAL, img_width, img_height);
		break;
	case KNN_FEATURE_1:
		adj = KNN_adj(s, features, 1, FEATURE, img_width, img_height);
		break;
	case KNN_FEATURE_2:
		adj = KNN_adj(s, features, 2, FEATURE, img_width, img_height);
		break;
	case KNN_FEATURE_4:
		adj = KNN_adj(s, features, 4, FEATURE, img_width, img_height);
		break;
	case KNN_FEATURE_8:
		adj = KNN_adj(s, features, 8, FEATURE, img_width, img_height);
		break;
	case KNN_FEATURE_16:
		adj = KNN_adj(s, features, 16, FEATURE, img_width, img_height);
        break;
    }

    PyArrayObject *edge_index;
    int64_t dims[2];
    dims[0] = 2;
    dims[1] = adj.size();
    edge_index = (PyArrayObject *) PyArray_SimpleNew(2, (npy_intp *)dims, PyArray_LONG);
    if (edge_index == NULL)
        return NULL;
    int i = 0;
    for(auto edge : adj)
    {
        *((long *)PyArray_GETPTR2(edge_index, 0, i)) = edge.first;
        *((long *)PyArray_GETPTR2(edge_index, 1, i)) = edge.second;
        i++;
    }
    return edge_index;
}

cv::Mat from_numpy(PyArrayObject *a)
{
    int ndims = PyArray_NDIM(a);
    int rows = PyArray_DIM(a, 0);
    int cols = PyArray_DIM(a, 1);

    cv::Mat img;
    if(ndims == 2)
    {
        img = cv::Mat(rows, cols, CV_32F);
        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
                img.at<float>(i, j) = *(float *)PyArray_GETPTR2(a, i, j) * 255;
    }
    else  // ndims == 3
    {
        img = cv::Mat(rows, cols, CV_32FC3);
        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
            {
                img.ptr<float>(i, j)[0] = *(float *)PyArray_GETPTR3(a, i, j, 0) * 255;
                img.ptr<float>(i, j)[1] = *(float *)PyArray_GETPTR3(a, i, j, 1) * 255;
                img.ptr<float>(i, j)[2] = *(float *)PyArray_GETPTR3(a, i, j, 2) * 255;
            }
    }
    return img;    
}

PyArrayObject *to_numpy_int32(cv::Mat a)
{
    PyArrayObject *x;
    int64_t x_dims[2];
    x_dims[0] = a.rows;
    x_dims[1] = a.cols;
    x = (PyArrayObject *) PyArray_SimpleNew(2, (npy_intp *)x_dims, PyArray_INT32);
    if(x == NULL)
        return NULL;
    for(int i=0; i<a.rows; i++)
        for(int j=0; j<a.cols; j++)
           *((int32_t *)PyArray_GETPTR2(x, i, j)) = a.at<int32_t>(i,j);
    return x;
}

PyArrayObject *to_numpy_float64(cv::Mat a)
{
    PyArrayObject *x;
    int64_t x_dims[a.rows];
    x_dims[0] = a.rows;
    x_dims[1] = a.cols;
    x = (PyArrayObject *) PyArray_SimpleNew(2, (npy_intp *)x_dims, PyArray_FLOAT64);
    if(x == NULL)
        return NULL;
    for(int i=0; i<a.rows; i++)
        for(int j=0; j<a.cols; j++)
           *((double *)PyArray_GETPTR2(x, i, j)) = a.at<double>(i,j);
    return x;
}

static PyObject* compute_features_color(PyObject *self, PyObject *args)
{
    PyArrayObject *img_np;
    GraphType graph_type;
    SegmentationMethod seg_method;
    int n_segments;
    float compactness;
    if(!PyArg_ParseTuple(args, "O!iii|f", &PyArray_Type, &img_np, &n_segments, &graph_type, &seg_method, &compactness))
        return NULL;

    cv::Mat img = from_numpy(img_np);
    cv::Mat img_cie_lab;
    cv::cvtColor(img, img_cie_lab, cv::COLOR_RGB2Lab);
    int region_size = sqrt((img.rows*img.cols)/n_segments);
    if (region_size < 2)
        region_size = 2;
    cv::Mat s;
    int slic_method = seg_method == ADAPTATIVE_SLIC ? cv::ximgproc::SLICO : cv::ximgproc::SLIC; 
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(img_cie_lab, slic_method, region_size);
    if (slic->getNumberOfSuperpixels() > 1)
    {
        slic->iterate();
        slic->enforceLabelConnectivity(25);
    }
    slic->getLabels(s);
    int n = slic->getNumberOfSuperpixels();

    cv::Mat features = color_features(s, n, img);

    PyArrayObject *features_np = to_numpy_float64(features);
    if (features_np == NULL)
        return NULL;

    PyArrayObject *edge_index = get_edge_index(s, features, graph_type, img.cols, img.rows);
    if (edge_index == NULL)
        return NULL;
    
    PyArrayObject *segments = to_numpy_int32(s);
    if(segments == NULL)
        return NULL;

    return Py_BuildValue("OOO", PyArray_Return(features_np), PyArray_Return(edge_index), PyArray_Return(segments));    
}

static PyObject* compute_features_gray(PyObject *self, PyObject *args)
{
    PyArrayObject *img_np;
    GraphType graph_type;
    SegmentationMethod seg_method;
    int n_segments;
    float compactness;
    if(!PyArg_ParseTuple(args, "O!iii|f", &PyArray_Type, &img_np, &n_segments, &graph_type, &seg_method, &compactness))
        return NULL;

    cv::Mat img = from_numpy(img_np);
    int region_size = sqrt((img.rows*img.cols)/n_segments);
    if (region_size < 2)
        region_size = 2;
    cv::Mat s;
    int slic_method = seg_method == ADAPTATIVE_SLIC ? cv::ximgproc::SLICO : cv::ximgproc::SLIC; 
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(img, slic_method, region_size);
    if (slic->getNumberOfSuperpixels() > 1)
    {
        slic->iterate();
        slic->enforceLabelConnectivity(25);
    }
    slic->getLabels(s);
    int n = slic->getNumberOfSuperpixels();

    cv::Mat features = grayscale_features(s, n, img);

    PyArrayObject *features_np = to_numpy_float64(features);
    if (features_np == NULL)
        return NULL;

    PyArrayObject *edge_index = get_edge_index(s, features, graph_type, img.cols, img.rows);
    if (edge_index == NULL)
        return NULL;

    PyArrayObject *segments = to_numpy_int32(s);
    if(segments == NULL)
        return NULL;

    return Py_BuildValue("OOO", PyArray_Return(features_np), PyArray_Return(edge_index), PyArray_Return(segments));    
}

static PyMethodDef compute_features_methods[] = {
    {"color_features", compute_features_color, METH_VARARGS, 
     "Computes features for RGB color datasets."},
    {"grayscale_features", compute_features_gray, METH_VARARGS, 
     "Computes features for grayscale datasets."}, 
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef compute_features_module = {
    PyModuleDef_HEAD_INIT, 
    "compute_features", 
    NULL, 
    -1,
    compute_features_methods
};

PyMODINIT_FUNC PyInit_compute_features(void)
{
    import_array();
    return PyModule_Create(&compute_features_module);
}