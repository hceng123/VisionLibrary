using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;


namespace EagleDefectDetector
{
    /// <summary>
    /// Follow steps 1a or 1b and then 2 to use this custom control in a XAML file.
    ///
    /// Step 1a) Using this custom control in a XAML file that exists in the current project.
    /// Add this XmlNamespace attribute to the root element of the markup file where it is 
    /// to be used:
    ///
    ///     xmlns:MyNamespace="clr-namespace:EagleDefectDetector"
    ///
    ///
    /// Step 1b) Using this custom control in a XAML file that exists in a different project.
    /// Add this XmlNamespace attribute to the root element of the markup file where it is 
    /// to be used:
    ///
    ///     xmlns:MyNamespace="clr-namespace:EagleDefectDetector;assembly=EagleDefectDetector"
    ///
    /// You will also need to add a project reference from the project where the XAML file lives
    /// to this project and Rebuild to avoid compilation errors:
    ///
    ///     Right click on the target project in the Solution Explorer and
    ///     "Add Reference"->"Projects"->[Browse to and select this project]
    ///
    ///
    /// Step 2)
    /// Go ahead and use your control in the XAML file.
    ///
    ///     <MyNamespace:CustomControl1/>
    ///
    /// </summary>
    public class VisionView : System.Windows.Controls.Image
    {
        private bool                        _bLeftMouseButtonDown = false;
        private Mat                         _mat;
        private Mat                         _learnMat;
        private System.Windows.Point        _leftClickStartPoint;
        private System.Windows.Point        _leftClickEndPoint;

        const double _maxMousePosX =        640;
        const double _maxMousePosY =        480;

        System.Drawing.Rectangle            _LearningROI;
        VectorOfKeyPoint                    _modelKeyPoints;
        VectorOfPoint                       _targetObjectPosition;
        private CONSTANTS.MACHINE_STATE     _machineState;
        private UMat                        _modelDescriptors;
        const double _hessianThresh =       300;

        public VisionView()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(VisionView), new FrameworkPropertyMetadata(typeof(VisionView)));
            _modelDescriptors = new UMat();
            _mat = null;
            _learnMat = null;
            _modelKeyPoints = new VectorOfKeyPoint();
            _targetObjectPosition = new VectorOfPoint();
        }

        [System.Runtime.InteropServices.DllImport("gdi32.dll")]
        static extern int DeleteObject(IntPtr o);

        protected static BitmapSource _loadBitmap(System.Drawing.Bitmap source)
        {
            IntPtr ip = source.GetHbitmap();
            BitmapSource bs = null;
            try
            {
                bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(ip,
                   IntPtr.Zero, Int32Rect.Empty,
                   System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());
            }
            finally
            {
                DeleteObject(ip);
            }

            return bs;
        }

        public void UpdateMat(Mat mat)
        {
            if ( null == _mat )
            {
                _mat = new Mat ( mat.Size, mat.Depth, mat.NumberOfChannels );
            }
            mat.CopyTo ( _mat );

            Mat displayMat = new Mat (mat, new System.Drawing.Rectangle(0, 0, mat.Width, mat.Height ) );      
            if ( CONSTANTS.MACHINE_STATE.LEARNING == _machineState )
            {
                System.Drawing.Point startPoint = new System.Drawing.Point(Convert.ToInt32(_leftClickStartPoint.X), Convert.ToInt32(_leftClickStartPoint.Y));
                System.Drawing.Size size = new System.Drawing.Size(Convert.ToInt32(_leftClickEndPoint.X - _leftClickStartPoint.X), Convert.ToInt32(_leftClickEndPoint.Y - _leftClickStartPoint.Y));
                System.Drawing.Rectangle rect = new System.Drawing.Rectangle(startPoint, size);

                _LearningROI = rect;

                MCvScalar scalar = new MCvScalar(255, 255, 255);

                CvInvoke.Rectangle( displayMat, rect, scalar);
            }
            else if ( CONSTANTS.MACHINE_STATE.AUTO == _machineState )
            {
                //if ( _targetObjectPosition.)
                CvInvoke.Polylines(displayMat, _targetObjectPosition, true, new MCvScalar(0, 255, 0, 255), 5);
            }

            IImage image = displayMat;
            Bitmap bitmap = image.Bitmap;
            this.Source = _loadBitmap(bitmap);            
        }

        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            Control src = e.Source as Control;
            System.Windows.Point pt = e.GetPosition(this);
            String strPosition = "X: " + pt.X.ToString() + " Y: " + pt.Y.ToString();
            switch (e.ChangedButton)
            {
                case MouseButton.Left:
                    _bLeftMouseButtonDown = true;
                    _leftClickStartPoint = e.GetPosition(this);
                    break;
                case MouseButton.Middle:
                    MessageBox.Show("Middle button");
                    break;
                case MouseButton.Right:
                    //MessageBox.Show("Right button");
                    break;
                default:
                    break;
            }
        }

        protected override void OnMouseUp(MouseButtonEventArgs e)
        {
            switch (e.ChangedButton)
            {
                case MouseButton.Left:
                    _bLeftMouseButtonDown = false;                 
                    break;
                case MouseButton.Middle:
                    MessageBox.Show("Middle button ");
                    break;
                case MouseButton.Right:
                    VisionOperation();
                    break;
                default:
                    break;
            }
        }

        protected void VisionOperation()
        {
            switch (_machineState)
            {
                case CONSTANTS.MACHINE_STATE.LEARNING:
                    Learn();
                    break;

                case CONSTANTS.MACHINE_STATE.AUTO:
                    FindObject();
                    Alignment();
                    break;

                default:
                    break;
            }
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            if ( _bLeftMouseButtonDown )
            {
                _leftClickEndPoint = e.GetPosition ( this );
            }
            base.OnMouseMove(e);
        }

        public CONSTANTS.MACHINE_STATE MachineState
        {
            get
            {
                return _machineState;
            }
            set
            {
                _machineState = value;
            }
        }

        public int Learn()
        {
            if ( _LearningROI.Width <= 0 || _LearningROI.Height <= 0 )
                return STATUS.NOK;
            if ( null == _learnMat )
            {
                _learnMat = new Mat(_mat.Size, _mat.Depth, _mat.NumberOfChannels );
            }
            _mat.CopyTo (_learnMat );

            CvInvoke.Imshow("Learning image", _learnMat );

            using (Mat modelMat = new Mat( _learnMat, _LearningROI))
            using (UMat uModelImage = modelMat.ToUMat(AccessType.Read))
            {
                SURF surfCPU = new SURF(_hessianThresh);
                surfCPU.DetectAndCompute ( modelMat, null, _modelKeyPoints, _modelDescriptors, false );
                if ( _modelDescriptors.IsEmpty )
                {
                    MessageBox.Show("Learn fail, not enough features");
                }
                else
                {
                    MessageBox.Show("Learn success!"); 
                }
            }
            string strMsg = string.Format("Learn ROI upper (left {0}, {1}), width {2}, height {3}", 
                _LearningROI.Left, _LearningROI.Top, _LearningROI.Width, _LearningROI.Height );
            Logger.Instance().AddText( strMsg );
            return STATUS.OK;
        }

        public int FindObject()
        {
            if (_modelDescriptors.IsEmpty)
            {
                MessageBox.Show("Please learn model first!");
                return STATUS.NOK;
            }
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            UMat observedDescriptors = new UMat();
            Mat homography = new Mat();
            int k = 2;
            double uniquenessThreshold = 0.8;
            VectorOfKeyPoint observedKeyPoints = new VectorOfKeyPoint();

            UMat uObservedImage = _mat.ToUMat(AccessType.Read);

            SURF surfCPU = new SURF(_hessianThresh);
            surfCPU.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);

            BFMatcher matcher = new BFMatcher(DistanceType.L2);
            matcher.Add( _modelDescriptors );

            matcher.KnnMatch(observedDescriptors, matches, k, null);
            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

            int nonZeroCount = CvInvoke.CountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(_modelKeyPoints, observedKeyPoints,
                   matches, mask, 1.5, 20);
                if (nonZeroCount >= 4)
                {
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(_modelKeyPoints,
                       observedKeyPoints, matches, mask, 2);

                    if (homography != null)
                    {
                        for (int row = 0; row < homography.Rows; ++row)
                        {
                            string szLine = "";
                            for (int col = 0; col < homography.Cols; ++col)
                            {
                                double dValue = homography.GetValue ( row, col );
                                szLine += dValue.ToString() + " ";
                            }
                            szLine += "\r\n";
                            Logger.Instance().AddText(szLine);
                        }

                        //draw a rectangle along the projected model
                        System.Drawing.Rectangle rect = new System.Drawing.Rectangle(System.Drawing.Point.Empty, _LearningROI.Size);
                        PointF[] pts = new PointF[]
                        {
                              new PointF(rect.Left, rect.Bottom),
                              new PointF(rect.Right, rect.Bottom),
                              new PointF(rect.Right, rect.Top),
                              new PointF(rect.Left, rect.Top)
                        };
                        pts = CvInvoke.PerspectiveTransform(pts, homography);

                        System.Drawing.Point[] points = Array.ConvertAll<PointF, System.Drawing.Point>(pts, System.Drawing.Point.Round);
                        _targetObjectPosition = new VectorOfPoint(points);

                        //CvInvoke.Polylines(_mat, _targetObjectPosition, true, new MCvScalar(0, 255, 0, 255), 5);
                    }
                }else
                {
                    MessageBox.Show("VoteForSizeAndOrientation not enough points");
                }
            }
            else
            {
                MessageBox.Show("VoteForUniqueness not enough unique features");
            }
            return STATUS.OK;
        }

        private void InitWarp(ref Mat W, float wz, float tx, float ty)
        {
            W.SetValue(0, 0, 1);
            W.SetValue(1, 0, wz);
            W.SetValue(2, 0, 0);

            W.SetValue(0, 1, -wz);
            W.SetValue(1, 1, 1);
            W.SetValue(2, 1, 0);

            W.SetValue(0, 2, tx);
            W.SetValue(1, 2, ty);
            W.SetValue(2, 2, 1);
        }

        void WarpImage(Mat src, ref Mat dst, Mat W)
        {
            dst.SetTo( new MCvScalar(0));

            Mat X = new Mat(3, 1, DepthType.Cv32F, 1 );
            Mat Z = new Mat(3, 1, DepthType.Cv32F, 1 );
            int x, y;

            for (x = 0; x < src.Width; x++)
            {
                for (y = 0; y < src.Height; y++)
                {
                    X.SetValue ( 0, 0, x );
                    X.SetValue ( 1, 0, y );
                    X.SetValue ( 2, 0, 1 );

                    CvInvoke.Gemm ( W, X, 1, null, 0, Z );

                    int x2, y2;
                    x2 = Convert.ToInt32 ( Z.GetValue(0, 0) );
                    y2 = Convert.ToInt32 ( Z.GetValue(1, 0) );

                    if (x2 >= 0 && x2 < dst.Width &&
                        y2 >= 0 && y2 < dst.Height)
                    {
                        double dValue = src.GetValue ( y, x );
                        dst.SetValue ( y2, x2, dValue );
                    }
                }
            }
        }

        private float Interpolate( Mat image, float x, float y)
        {
            int xi = Convert.ToInt32 ( Math.Floor(x) );
            int yi = Convert.ToInt32 ( Math.Floor(y) );

            float k1 = x - xi;
            float k2 = y - yi;

            bool b1 = xi < image.Width - 1;
            bool b2 = yi < image.Height - 1;

            float fUL = (float)image.GetValue ( yi, xi );
            float fUR = (float) ( b1 ? image.GetValue ( yi, xi + 1 ) : 0 );
            float fLL = (float) ( b2 ? image.GetValue ( yi + 1, xi ) : 0 );
            float fLR = (float) ( b1 && b2 ? image.GetValue ( yi + 1, xi + 1 ) : 0 );

            float interpolatedValue = ( 1.0f- k1) * ( 1.0f - k2 ) * fUL +
                k1*(1.0f-k2)*fUR + (1.0f-k1)*k2*fLL + k1*k2*fLR;
            return interpolatedValue;
        }

        public int Alignment()
        {
            const float EPS = 1E-5f; // Threshold value for termination criteria.
            const int MAX_ITER = 50;  // Maximum iteration count.
            bool bSuccess = false;
            string szLine = "";
            
            //Mat inputMat = _mat;

            System.Drawing.Size size = _mat.Size;
            Mat gradIx = new Mat( size, DepthType.Cv16S, 1);
            Mat gradIy = new Mat( size, DepthType.Cv16S, 1);            

            Mat W = new Mat(3, 3, DepthType.Cv32F, 1);
            Mat X = new Mat(3, 1, DepthType.Cv32F, 1);
            Mat Z = new Mat(3, 1, DepthType.Cv32F, 1);

            Mat H = new Mat(3, 3, DepthType.Cv32F, 1);
            Mat iH = new Mat(3, 3, DepthType.Cv32F, 1);
            Mat b = new Mat(3, 1, DepthType.Cv32F, 1);
            Mat deltaP = new Mat(3, 1, DepthType.Cv32F, 1);

            CvInvoke.Imshow ( "Learn Image", _learnMat );

            Mat grayLearnMat = new Mat ( _learnMat.Size, DepthType.Cv8U, 1 );
            CvInvoke.CvtColor ( _learnMat, grayLearnMat, ColorConversion.Bgr2Gray );
            CvInvoke.Imshow ( "Gray Learn Image", grayLearnMat );

            InitWarp(ref W, -0.05f, 5, 5);
            Mat inputMat = new Mat( grayLearnMat.Size, grayLearnMat.Depth, 1 );
            CvInvoke.CvtColor ( _mat, inputMat, ColorConversion.Bgr2Gray );
            //WarpImage ( grayLearnMat, ref inputMat, W );
            CvInvoke.Imshow ( "Input Image", inputMat );

            //return STATUS.OK;
            
            CvInvoke.Sobel(inputMat, gradIx, inputMat.Depth, 1, 0);
            CvInvoke.ConvertScaleAbs ( gradIx, gradIx, 0.125, 0 );
            CvInvoke.Sobel(inputMat, gradIy, inputMat.Depth, 0, 1);
            CvInvoke.ConvertScaleAbs ( gradIy, gradIy, 0.125, 0 );

            // Here we will store parameter approximation. 
            float wz_a = 0, tx_a = 0, ty_a = 0;

            // Here we will store current mean error value.
            float mean_error = 0;

            // Iterate
            int iter = 0; // number of current iteration
            while (iter < MAX_ITER)
            {
                ++iter;
                mean_error = 0;
                int pixelCount = 0;
                InitWarp(ref W, wz_a, tx_a, ty_a);

                H.SetTo(new MCvScalar(0));
                b.SetTo(new MCvScalar(0));

                // (u,v) - pixel coordinates in the coordinate frame of T.
                int u, v;

                // Walk through pixels in the template T.
                int i, j;

                for (i = 0; i < _LearningROI.Width; i++)
                {
                    u = i + _LearningROI.X;

                    for (j = 0; j < _LearningROI.Height; j++)
                    {
                        v = j + _LearningROI.Y;

                        X.SetValue(0, 0, u);
                        X.SetValue(1, 0, v);
                        X.SetValue(2, 0, 1);

                        CvInvoke.Gemm(W, X, 1, null, 0, Z );

                        float u2 = (float)Z.GetValue ( 0, 0 );
                        float v2 = (float)Z.GetValue ( 1, 0 );

                        int u2i = Convert.ToInt32 ( Math.Floor ( u2 ) );
                        int v2i = Convert.ToInt32 ( Math.Floor ( v2 ) );

                        if ( u2i >= 0 && u2i < inputMat.Width
                            && v2i >= 0 && v2i < inputMat.Height )
                        {
                            ++ pixelCount;

                            float Ix = Interpolate(gradIx, u2, v2);
                            float Iy = Interpolate(gradIy, u2, v2);

                            // Calculate steepest descent image's element.
                            //stdesc = ∇I * ∂W / ∂p
                            float []stdesc = new float[3]; // an element of steepest descent image
                            stdesc[0] = (float)(-v * Ix + u * Iy);
                            stdesc[1] = (float)Ix;
                            stdesc[2] = (float)Iy;

                            float I2 = Interpolate ( inputMat, u2, v2 );

                            float D = (float)grayLearnMat.GetValue ( v, u ) - I2;
                            mean_error += Math.Abs(D);

                            float[] pb = new float[3];
                            pb[0] = (float) b.GetValue ( 0, 0 );
                            pb[1] = (float) b.GetValue ( 1, 0 );
                            pb[2] = (float) b.GetValue ( 2, 0 );

                            pb[0] += stdesc[0] * D;
                            pb[1] += stdesc[1] * D;
                            pb[2] += stdesc[2] * D;

                            b.SetValue(0, 0, pb[0]);
                            b.SetValue(1, 0, pb[1]);
                            b.SetValue(2, 0, pb[2]);

                            int l, m;
                            for (l = 0; l < 3; ++ l )
                            {
                                for (m = 0; m < 3; ++ m)
                                {
                                    double dTemp = H.GetValue ( l, m );
                                    dTemp += stdesc[l] * stdesc[m];
                                    H.SetValue ( l, m, dTemp );
                                }
                            }
                        }
                    }
                }

                if ( pixelCount > 0 )
                    mean_error /= pixelCount;
                double inv_res = CvInvoke.Invert(H, iH, DecompMethod.LU);
                if (inv_res == 0)
                {
                    return STATUS.NOK;
                }

                CvInvoke.Gemm(iH, b, 1, null, 0, deltaP);

                float delta_wz = (float)deltaP.GetValue ( 0, 0);
                float delta_tx = (float)deltaP.GetValue ( 1, 0);
                float delta_ty = (float)deltaP.GetValue ( 2, 0);

                wz_a += delta_wz;
                tx_a += delta_tx;
                ty_a += delta_ty;

                szLine = string.Format("iter = {0} dwz = {1} dtx = {2} dty = {3} mean_error = {4}",
                                              iter, delta_wz, delta_tx, delta_ty, mean_error);
                Logger.Instance().AddText(szLine);

                // Check termination critera.
                if (Math.Abs(delta_wz) < EPS && Math.Abs(delta_tx) < EPS && Math.Abs(delta_ty) < EPS )
                {
                    bSuccess = true;
                    break;
                }        
            }

            Logger.Instance().AddText("===============================================");
            Logger.Instance().AddText("Algorithm: forward additive");
            
            Logger.Instance().AddText(string.Format("Iteration count: {0}", iter));
            Logger.Instance().AddText(string.Format("Approximation: wz_a= {0} tx_a={1} ty_a={2}", wz_a, tx_a, ty_a));
            Logger.Instance().AddText(string.Format("Epsilon: {0}", EPS));
            Logger.Instance().AddText(string.Format("Resulting mean error: {0}", mean_error));
            Logger.Instance().AddText("===============================================");

            if (bSuccess)
                return STATUS.OK;
            else
                return STATUS.NOK;
        }
    }
}
