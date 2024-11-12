/****************************************************************************/
//
// Matlab interface file:  smo_FAN.cpp
//
// Written 25/01/2024 by F. DUFRENOIS.
// Fits a part-structured model to a binary image.
// Uses bicubic interpolation for subpixel accuracy on translation.
//
// [alpha,b,w] = smoFan_mex(K,y,beta,C)
//
/****************************************************************************/

#include <math.h>
#include <time.h>
#include <stdio.h>      // printf, scanf, puts, NULL */
#include <stdlib.h>   //rand srand
#include "mymex.h"


#define LARGE 1e10
#define EPS 1e-3
#define TAU 1e-12
#define ITERMAX 10000 


 //return (Y(i) * E < - epsilon && alpha(i) < C) || (Y(i) * E > epsilon && alpha(i) > 0);
bool Iup(double a, short y, double C){
    return (y ==1 && a < C) || (y==-1 && a > 0);
}
bool Idown(double a, short y, double C){
    return (y ==1 && a >0) || (y==-1 && a < C);
}
// WorkingSetSelection(i,j,K,alpha,y,grad,m,C);
double WorkingSetSelection(int &i,int &j, double *K,double *a,short *y, double *grad,int m,double C){

double Gmax,Gmin,objmin,r,aa,b,delta;
int k;
Gmax=-mxGetInf();
Gmin=mxGetInf();
i=-1;
for (k=0;k<m;k++){
    if (Iup(a[k],y[k],C)){
        if ((-y[k]*grad[k])>=Gmax){
            i=k;
            Gmax=-y[k]*grad[k];
        }
    }
}
j=-1;
objmin=mxGetInf();
for (k=0;k<m;k++){
    if (Idown(a[k],y[k],C)){
        
        if ((-y[k]*grad[k])<=Gmin){
            Gmin=-y[k]*grad[k];
        }
        b=Gmax+(y[k]*grad[k]);
        if(b>0){
            aa=K[i+i*m]+K[k+k*m]-2*y[i]*y[k]*K[i+k*m];
            if (aa<=0){aa=TAU;}
            r=-b*b/aa;
            if (r<=objmin){
                j=k;
                objmin=r;
            }
        }
    }
}
delta=Gmax-Gmin;
if (delta<EPS){
    i=-1;
    j=-1;
}
return (delta);
}


// updt_w(i1,i2,K,m,t1,t2,b - b_,w);
void UpdGrad(double *grad,int i,int j, double *K,short *y,double ti,double tj,int m){
int k;
    for(k=0;k<m;k++)
        grad[k] += y[k]*(ti*K[i+k*m]+tj*K[j+k*m]);
    
}

double BuildBias(double *K,double *a,short *y, double *beta,int m){
    int k,i;
    double res,bias=0.0;
    //allbias=(double *)mxCalloc(m,sizeof(double));
    //res=(double *)mxCalloc(m,sizeof(double));
    for(i=0;i<m;i++){
         res=0.0;
        for(k=0;k<m;k++){
            if (a[k]>0)
                res+=y[k]*a[k]*K[i+k*m];
        }
        bias+=beta[i]*y[i]-res;
    }
    return (bias/m);
}
    
//updt_error(i1,i2,K,m,t1,t2,alpha,C,error);
void proj(double *output,double *K,double *a,double *b,short *y,int m){
    int k,i;
    double res;
    
    for(i=0;i<m;i++){
        res=0.0;
        for(k=0;k<m;k++){
            if (a[k]>0)
                res+=y[k]*a[k]*K[i+k*m];
        }
        output[i]=res+(*b);
    }
    
}



/****************************************************************************/
//
// gateway driver to call the distance transform code from matlab
//
// This is the matlab entry point
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int nrow_K, ncol_K, m,i,j,k;
  double *K,*beta,C; //input
  short *y;  // input
  double *alpha,*w,*bias,*grad; //output
  int *iter;
  double a,b,ai_old,aj_old,S,ti,tj;

  //------------- INPUT;
  // read kernel gram matrix: double
  nrow_K = (int)mxGetM(prhs[0]);//mexPrintf("%d ",nrow);
  ncol_K = (int)mxGetN(prhs[0]);// mexPrintf("%d ",ncol);
  errCheck(nrow_K==ncol_K,"kernel Gram matrix must be square!");
  K=mxGetPr(prhs[0]);  //printMatrow(K,nrow_K,10,1); //  printKrow(K,nrow,ncol,2);  printKcol(K,nrow,1);


 //label vector: short (int16 in matlab)
  m=(int)mxGetM(prhs[1]);
  y=(short*)mxGetPr(prhs[1]);     //y = (short *)mxGetData(prhs[1]);
  // beta
  beta=mxGetPr(prhs[2]);
  // read C: double
  C= mxGetScalar(prhs[3]);


  //------------- OUTPUT;
  // return alpha
  plhs[0]=mxCreateDoubleMatrix(m,1,mxREAL);
  alpha=mxGetPr(plhs[0]);
  // return b
  plhs[1]=mxCreateDoubleMatrix(1,1,mxREAL);
  bias=mxGetPr(plhs[1]);// bias
  // return w
  plhs[2]=mxCreateDoubleMatrix(m,1,mxREAL);
  w=mxGetPr(plhs[2]);//proj
  plhs[3]=mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
  iter=(int*)mxGetData(plhs[3]);
  //------- AUXILIAR variables
  // define gradient vector and initialization with -beta
  grad=(double *)mxCalloc(m,sizeof(double));
  for (k=0;k<m;k++){grad[k]=-beta[k];}
  *iter=0;
  //train smo_FAN
  while (*iter<ITERMAX){
       WorkingSetSelection(i,j,K,alpha,y,grad,m,C);
        //mexPrintf("[%d - %d - %d] \n ",i,j,iter);
       if (j==-1){break;}
       
       a = K[i+i*m]+K[j+j*m]-2*y[i]*y[j]*K[i+j*m];
       if (a<=0) {a=TAU;}
       b=-y[i]*grad[i]+y[j]*grad[j];
       
       ai_old=alpha[i];
       aj_old=alpha[j];
       S=y[i]*ai_old+y[j]*aj_old;
        
       // update alpha_i 
       alpha[i]+=y[i]*(b/a);
       alpha[i]=CLIP(alpha[i],0.,C);
       
       //update alphaj
      
       alpha[j]=y[j]*(S-y[i]*alpha[i]);
       alpha[j]=CLIP(alpha[j],0.,C);
    
        //update alphai
        alpha[i]=y[i]*(S-y[j]*alpha[j]);
        
        ti=y[i]*(alpha[i]-ai_old);
        tj=y[j]*(alpha[j]-aj_old);
        // update gradient
        UpdGrad(grad,i,j,K,y,ti,tj,m);
        
        //increment iter
        *iter=*iter+1;
  }
  
  for (k=0;k<m;k++){
      if (alpha[k]<1e-10)
          alpha[k]=0.;
  }
  //build bias
  *bias=BuildBias(K,alpha,y,beta,m);
  // compute proj
  proj(w,K,alpha,bias,y,m);
  // free memory
  mxFree(grad);
}
