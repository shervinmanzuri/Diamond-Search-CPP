#ifndef IMAGEMATCHER_DIAMONDSEARCH_H
#define IMAGEMATCHER_DIAMONDSEARCH_H

#include "../ds_search_result.h"
#include <opencv2/core/core.hpp>
#include <unordered_map>
#include <map>

enum COST_FUNCTION { FUNC_PSNR = 0, FUNC_MAD };

DS_Search_Result
motionEstDS(cv::Mat_<int> &currImg, cv::Mat_<int> &refImg, int &mbSize, int &p, COST_FUNCTION &c, int skip);

#endif //IMAGEMATCHER_DIAMONDSEARCH_H
