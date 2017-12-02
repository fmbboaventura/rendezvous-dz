__kernel void brute_Jn(
   __global float* X,
   __global float* ve,
   __global float* gama,
   __global float* w,
   __global float* jn,
   const unsigned int count_gama,
   const unsigned int count_ve,
   const unsigned int count_X
)
{
   int i_gama = get_global_id(2);
   int i_ve = get_global_id(1);
   int i_X = get_global_id(0);
   int n;
   float temp;
   float gama_i = gama[i_gama];
   float ve_i = ve[i_ve];
   float X_i = X[i_X];
   if(i_X < count_X && i_ve < count_ve && i_gama < count_gama){
       for (n=1; n <= 20; n++){
           temp = (pow(1,(float)(n+1)))/
                  ((*w)*n*pow(X_i, (float)n))/
                  (1+(n*gama_i/(*w))*(n*gama_i/(*w)));
           if (n%2==0) temp = -temp;
           jn[i_X*count_ve*count_gama*20 +
               i_ve*count_gama*20 +
               i_gama*20 + (n-1)] = ve_i * temp;
       }
   }
}
