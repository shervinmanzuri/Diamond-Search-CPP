#include "diamondsearch.h"
#include "../arrayhash.h"
#include "../Metrics/mad.h"
#include "../Metrics/psnr.h"

#include <vector>
#include <map>
#include <cstdlib>
#include <unordered_map>

using namespace std;

float costFunc(cv::Mat_<int> &currentBlk, cv::Mat_<int> &refBlk, int &n, COST_FUNCTION &c) {
    switch (c)
    {
        case FUNC_PSNR:
            return 361.202 - cv::PSNR(currentBlk, refBlk);
        case FUNC_MAD:
            return getMAD(currentBlk, refBlk, n);
        default:
            return 361.202 - cv::PSNR(currentBlk, refBlk);
    }
}

bool checkBlock(int &refBlkVer, int &refBlkHor, int &row, int &col, int &mbSize) {
    return refBlkVer < 0 || refBlkVer + mbSize > row || refBlkHor < 0 || refBlkHor + mbSize > col;
}

bool checkBlock2(int &refBlkVer, int &refBlkHor, int &j, int &i, int &p) {
    return refBlkHor < j - p || refBlkHor > j + p || refBlkVer < i - p || refBlkVer > i + p;
}

DS_Search_Result
motionEstDS(cv::Mat_<int> &currImg, cv::Mat_<int> &refImg, int &mbSize, int &p, COST_FUNCTION &c, int skip) {

    std::map< std::pair<int,int>, std::pair<int,int>> mv;

    int row = currImg.rows;
    int col = currImg.cols;

    std::vector<float> costs(9, 65537);

    std::vector<cv::Point2i> LDSP{
            cv::Point2i(0, -2),
            cv::Point2i(-1, -1),
            cv::Point2i(1, -1),
            cv::Point2i(-2, 0),
            cv::Point2i(0, 0),
            cv::Point2i(2, 0),
            cv::Point2i(-1, 1),
            cv::Point2i(1, 1),
            cv::Point2i(0, 2),
    };
    std::vector<cv::Point2i> SDSP{
            cv::Point2i(0, -1),
            cv::Point2i(-1, 0),
            cv::Point2i(0, 0),
            cv::Point2i(1, 0),
            cv::Point2i(0, 1),
    };

    int computations = 0;

    std::unordered_map< std::array<int, 4>, float, container_hasher> ds_computations;

    int mbCount = 0;
    int x, y;

    //TODO implement k skip
    for (int i = 0; i <= row - mbSize; i = i + (mbSize * skip)) {
        for (int j = 0; j <= col - mbSize; j = j + (mbSize * skip)) {
            x = j;
            y = i;
            cv::Mat_ currentBlk = currImg(cv::Range(i, i + mbSize), cv::Range(j, j + mbSize));
            cv::Mat_ refBlk = refImg(cv::Range(i, i + mbSize), cv::Range(j, j + mbSize));
            costs[4] = costFunc(currentBlk, refBlk, mbSize, c);
            ds_computations[std::array<int,4>{j, i, j+mbSize, i+mbSize}] = costs[4];
            computations++;

            for (int k = 0; k < 9; k++) {
                int refBlkVer = y + LDSP[k].y;
                int refBlkHor = x + LDSP[k].x;
                if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize) || k == 4) {
                    continue;
                }
                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize), cv::Range(refBlkHor, refBlkHor + mbSize));
                costs[k] = costFunc(currentBlk, refBlk, mbSize, c);
                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[k];
                computations++;
            }

            int point = std::min_element(costs.begin(), costs.end()) - costs.begin();
            float cost = costs[point];

            int cornerFlag = 0;
            int xLast = i, yLast = j;
            int SDSPFlag;
            if (point == 4) {
                SDSPFlag = 1;
            } else {
                SDSPFlag = 0;
                if (std::abs(LDSP[point].x) == std::abs(LDSP[point].y)) {
                    cornerFlag = 0;
                } else {
                    cornerFlag = 1;
                }
                xLast = x;
                yLast = y;
                x += LDSP[point].x;
                y += LDSP[point].y;
                std::fill(costs.begin(), costs.end(), 65537);
                costs[4] = cost;
            }


            while (SDSPFlag == 0) {
                if (cornerFlag == 1) {
                    for (int k = 0; k < 9; k++) {
                        int refBlkVer = y + LDSP[k].y;
                        int refBlkHor = x + LDSP[k].x;
                        if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize) || k == 4) {
                            continue;
                        }
                        if (refBlkHor >= xLast - 1 && refBlkHor <= xLast + 1 && refBlkVer >= yLast - 1 &&
                            refBlkVer <= yLast + 1) {
                            continue;
                        } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                            continue;
                        } else {
                            refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                            cv::Range(refBlkHor, refBlkHor + mbSize));
                            costs[k] = costFunc(currentBlk, refBlk, mbSize, c);
                            ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[k];
                            computations++;
                        }
                    }
                } else {
                    switch (point) {
                        case 1: {
                            int refBlkVer = y + LDSP[0].y;
                            int refBlkHor = x + LDSP[0].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[0] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[0];
                                computations++;
                            }
                            refBlkVer = y + LDSP[1].y;
                            refBlkHor = x + LDSP[1].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[1] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[1];
                                computations++;
                            }
                            refBlkVer = y + LDSP[3].y;
                            refBlkHor = x + LDSP[3].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[3] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[3];
                                computations++;
                            }
                            break;
                        }
                        case 2: {
                            int refBlkVer = y + LDSP[0].y;
                            int refBlkHor = x + LDSP[0].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[0] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[0];
                                computations++;
                            }
                            refBlkVer = y + LDSP[2].y;
                            refBlkHor = x + LDSP[2].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[2] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[2];
                                computations++;
                            }
                            refBlkVer = y + LDSP[5].y;
                            refBlkHor = x + LDSP[5].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[5] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[5];
                                computations++;
                            }
                            break;
                        }
                        case 6: {
                            int refBlkVer = y + LDSP[3].y;
                            int refBlkHor = x + LDSP[3].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[3] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[3];
                                computations++;
                            }
                            refBlkVer = y + LDSP[6].y;
                            refBlkHor = x + LDSP[6].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[6] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[6];
                                computations++;
                            }
                            refBlkVer = y + LDSP[8].y;
                            refBlkHor = x + LDSP[8].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[8] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[8];
                                computations++;
                            }
                            break;
                        }
                        case 7: {
                            int refBlkVer = y + LDSP[5].y;
                            int refBlkHor = x + LDSP[5].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[5] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[5];
                                computations++;
                            }
                            refBlkVer = y + LDSP[7].y;
                            refBlkHor = x + LDSP[7].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[7] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[7];
                                computations++;
                            }
                            refBlkVer = y + LDSP[8].y;
                            refBlkHor = x + LDSP[8].x;
                            if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                                // Do Nothing!
                            } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                                // Do Nothing!
                            } else {
                                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                                cv::Range(refBlkHor, refBlkHor + mbSize));
                                costs[8] = costFunc(currentBlk, refBlk, mbSize, c);
                                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[8];
                                computations++;
                            }
                            break;
                        }
                        default: {
                            break;
                        }
                    }
                }
                point = std::min_element(costs.begin(), costs.end()) - costs.begin();
                cost = costs[point];
                if (point == 4) {
                    SDSPFlag = 1;
                } else {
                    SDSPFlag = 0;
                    if (std::abs(LDSP[point].x) == std::abs(LDSP[point].y)) {
                        cornerFlag = 0;
                    } else {
                        cornerFlag = 1;
                    }
                    xLast = x;
                    yLast = y;
                    x += LDSP[point].x;
                    y += LDSP[point].y;
                    std::fill(costs.begin(), costs.end(), 65537);
                    costs[4] = cost;
                }
            }

            std::fill(costs.begin(), costs.end(), 65537);
            costs[2] = cost;
            for (int k = 0; k < 5; k++) {
                int refBlkVer = y + SDSP[k].y;
                int refBlkHor = x + SDSP[k].x;
                if (checkBlock(refBlkVer, refBlkHor, row, col, mbSize)) {
                    continue;
                } else if (checkBlock2(refBlkVer, refBlkHor, j, i, p)) {
                    continue;
                }
                if (k == 2) {
                    continue;
                }
                refBlk = refImg(cv::Range(refBlkVer, refBlkVer + mbSize),
                                cv::Range(refBlkHor, refBlkHor + mbSize));
                costs[k] = costFunc(currentBlk, refBlk, mbSize, c);
                ds_computations[std::array<int,4>{refBlkHor, refBlkVer, refBlkHor+mbSize, refBlkVer+mbSize}] = costs[k];
                computations++;
            }

            point = std::min_element(costs.begin(), costs.end()) - costs.begin();


            x += SDSP[point].x;
            y += SDSP[point].y;

            ds_computations[std::array<int,4>{j, i, x, y}] = costs[point];

            // y - i;
            // x - j;
            mbCount++;

            mv.insert({{j,i},{x,y}});

            std::fill(costs.begin(), costs.end(), 65537);
        }
    }

    DS_Search_Result result = {ds_computations, mv};
    return result;
}