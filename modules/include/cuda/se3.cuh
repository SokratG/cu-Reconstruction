#ifndef CUPHOTO_LIB_SE3_CUH
#define CUPHOTO_LIB_SE3_CUH

#include "CudaUtils/cudaUtility.cuh"
#include "CudaUtils/cudaMath.cuh"

#include "CudaUtils/logging.h"

#include "types.cuh"


namespace cuphoto {

template<typename Type>
class SE3
{
public:
    __device__ SE3() {
        se3[0][0] = 1; se3[0][1] = 0; se3[0][2] = 0; se3[0][3] = 0;
        se3[1][0] = 0; se3[1][1] = 1; se3[1][2] = 0; se3[1][3] = 0;
        se3[2][0] = 0; se3[2][1] = 0; se3[2][2] = 1; se3[2][3] = 0;
    }

    __device__ SE3(const Type qw, const Type qx, const Type qy, const Type qz, 
                   const Type tx, const Type ty, const Type tz) {
        const Type x  = 2*qx;
        const Type y  = 2*qy;
        const Type z  = 2*qz;
        const Type wx = x*qw;
        const Type wy = y*qw;
        const Type wz = z*qw;
        const Type xx = x*qx;
        const Type xy = y*qx;
        const Type xz = z*qx;
        const Type yy = y*qy;
        const Type yz = z*qy;
        const Type zz = z*qz;

        init(wx, wy, wz, xx, xy, xz, yy, yz, zz, tx, ty, tz);
    }

    __device__ SE3(const float4& qaut, const float3& trans) {
        const Type x  = 2*qaut.x;
        const Type y  = 2*qaut.y;
        const Type z  = 2*qaut.z;
        const Type wx = x*qaut.w;
        const Type wy = y*qaut.w;
        const Type wz = z*qaut.w;
        const Type xx = x*qaut.x;
        const Type xy = y*qaut.x;
        const Type xz = z*qaut.x;
        const Type yy = y*qaut.y;
        const Type yz = z*qaut.y;
        const Type zz = z*qaut.z;

        init(wx, wy, wz, xx, xy, xz, yy, yz, zz, trans.x, trans.y, trans.z);
    }

    __device__ SE3(const Type r[9], const Type t[3]) {
        se3[0][0]=r[0]; se3[0][1]=r[1]; se3[0][2]=r[2]; se3[0][3]=t[0];
        se3[1][0]=r[3]; se3[1][1]=r[4]; se3[1][2]=r[5]; se3[1][3]=t[1];
        se3[2][0]=r[6]; se3[2][1]=r[7]; se3[2][2]=r[8]; se3[2][3]=t[2];
    }

    __device__ SE3<Type> inv() const {
        SE3<Type> result;
        result.se3[0][0] = se3[0][0];
        result.se3[0][1] = se3[1][0];
        result.se3[0][2] = se3[2][0];
        result.se3[1][0] = se3[0][1];
        result.se3[1][1] = se3[1][2];
        result.se3[1][2] = se3[2][1];
        result.se3[2][0] = se3[0][2];
        result.se3[2][1] = se3[1][2];
        result.se3[2][2] = se3[2][2];
        result.se3[0][3] = -se3[0][0]*se3[0][3]-se3[1][0]*se3[1][3]-se3[2][0]*se3[2][3];
        result.se3[1][3] = -se3[0][1]*se3[0][3]-se3[1][2]*se3[1][3]-se3[2][1]*se3[2][3];
        result.se3[2][3] = -se3[0][2]*se3[0][3]-se3[1][3]*se3[1][3]-se3[2][2]*se3[2][3];
        return result;
    }

    __device__ float3 rotate(const float3& p) const {
        return make_float3(se3[0][0]*p.x + se3[0][1]*p.y + se3[0][2]*p.z,
                           se3[1][0]*p.x + se3[1][1]*p.y + se3[1][2]*p.z,
                           se3[2][0]*p.x + se3[2][1]*p.y + se3[2][2]*p.z);
    }

    __device__ float3 translate(const float3& p) const {
        return make_float3(p.x + se3[0][3],
                           p.y + se3[1][3],
                           p.z + se3[2][3]);
    }

    __device__ float3 operator*(const float3& p) {
        return translate(rotate(p));
    }

    __device__ inline ui64 get_size() const { return 12 * sizeof(Type); }

    __device__ inline Type operator()(const i32 row, const i32 col) const {
        if (row >= 3 || row < 0 || col >= 4 || col < 0) {
            LogError(LOG_CUDA "SE3::operator() -- index is out of range (%i, %i)\n", row, col);
        }
        
        return se3[row][col];
    }

    __device__ inline Type& operator()(const i32 row, const i32 col) {
        if (row >= 3 || row < 0 || col >= 4 || col < 0) {
            LogError(LOG_CUDA "SE3::operator() -- index is out of range (%i, %i)\n", row, col);
        }
        
        return se3[row][col];
    }

    __device__ inline Type* data() const { 
        return se3; 
    }
private:
    __device__ inline void init(const Type wx, const Type wy, const Type wz, const Type xx,
                                const Type xy, const Type xz, const Type yy, const Type yz, 
                                const Type zz, const Type tx, const Type ty, const Type tz) {
        se3[0][0] = 1-(yy+zz);
        se3[0][1] = xy-wz;
        se3[0][2] = xz+wy;
        se3[1][0] = xy+wz;
        se3[1][1] = 1-(xx+zz);
        se3[1][2] = yz-wx;
        se3[2][0] = xz-wy;
        se3[2][1] = yz+wx;
        se3[2][2] = 1-(xx+yy);

        se3[0][3] = tx;
        se3[1][3] = ty;
        se3[2][3] = tz;
    }

    Type se3[3][4];
};

template<typename Type>
__device__ SE3<Type> operator*(const SE3<Type> &lhs, const SE3<Type> &rhs) {
    SE3<Type> result;

    result(0, 0)  = lhs(0, 0)*rhs(0, 0) + lhs(0, 1)*rhs(1, 0) + lhs(0, 2)*rhs(2, 0);
    result(0, 1)  = lhs(0, 0)*rhs(0, 1) + lhs(0, 1)*rhs(1, 1) + lhs(0, 2)*rhs(2, 1);
    result(0, 2)  = lhs(0, 0)*rhs(0, 2) + lhs(0, 1)*rhs(1, 2) + lhs(0, 2)*rhs(2, 2);
    result(0, 3)  = lhs(0, 3) + lhs(0, 0)*rhs(0, 3) + lhs(0, 1)*rhs(1, 3) + lhs(0, 2)*rhs(2, 3);
    result(1, 0)  = lhs(1, 0)*rhs(0, 0) + lhs(1, 1)*rhs(1, 0) + lhs(1, 2)*rhs(2, 0);
    result(1, 1)  = lhs(1, 0)*rhs(0, 1) + lhs(1, 1)*rhs(1, 1) + lhs(1, 2)*rhs(2, 1);
    result(1, 2)  = lhs(1, 0)*rhs(0, 2) + lhs(1, 1)*rhs(1, 2) + lhs(1, 2)*rhs(2, 2);
    result(1, 3)  = lhs(1, 3) + lhs(1, 0)*rhs(0, 3) + lhs(1, 1)*rhs(1, 3) + lhs(1, 2)*rhs(2, 3);
    result(2, 0)  = lhs(2, 0)*rhs(0, 0) + lhs(2, 1)*rhs(1, 0) + lhs(2, 2)*rhs(2, 0);
    result(2, 1)  = lhs(2, 0)*rhs(0, 1) + lhs(2, 1)*rhs(1, 1) + lhs(2, 2)*rhs(2, 1);
    result(2, 2) = lhs(2, 0)*rhs(0, 2) + lhs(2, 1)*rhs(1, 2) + lhs(2, 2)*rhs(2, 2);
    result(2, 3) = lhs(2, 3) + lhs(2, 0)*rhs(0, 3) + lhs(2, 1)*rhs(1, 3) + lhs(2, 2)*rhs(2, 3);

    return result;
}


};




#endif