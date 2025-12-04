// هذا هو ملف الـ backend الرئيسي
// ببساطة: هو اللي هيستقبل الرسائل من الـ form ويحفظها

const express = require("express");
const cors = require("cors");
const nodemailer = require("nodemailer");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 5000;



// بننشئ transporter للـ email (مثل البريد اللي هيرسل الرسالة)
// لو الإيميل مش مضبوط في .env، الـ transporter مش هيشتغل (هيفشل لما نحاول نرسل)
const transporter = nodemailer.createTransport({
  service: "gmail", // ممكن تستخدم "outlook" أو "yahoo" كمان
  auth: {
    user: process.env.EMAIL_USER, // الإيميل اللي هيرسل منه (من ملف .env)
    pass: process.env.EMAIL_PASS, // App Password (من ملف .env)
  },
  tls: {
    rejectUnauthorized: false, // عشان نتجاوز مشكلة SSL certificate
  },
});


// ده بيسمح للـ frontend (React) يرسل طلبات للـ backend
app.use(cors());
app.use(express.json()); // ده عشان نفهم الـ JSON اللي جاي من الـ frontend

// ده الـ endpoint (عنوان) اللي هيستقبل رسائل الـ contact form
app.post("/api/contact", async (req, res) => {
  try {
    // بناخد البيانات اللي جاية من الـ form
    const { name, email, message } = req.body;

    // بنتحقق إن كل الحقول موجودة
    if (!name || !email || !message) {
      return res.status(400).json({
        success: false,
        message: "fill_fields", // كود الترجمة (الـ frontend هيتولى الترجمة)
      });
    }

    // بنرسل الإيميل
    const mailOptions = {
      from: process.env.EMAIL_USER, // لازم يكون نفس إيميل المصادقة (Gmail مش بيسمح بغير كده)
      to: process.env.EMAIL_USER, // للإيميل بتاعك (ممكن تغيره لإيميل تاني)
      replyTo: email, // عشان لما ترد، الرد يروح لإيميل المستخدم
      subject: `New message from ${name}`, // عنوان الإيميل
      html: `
        <div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5;">
          <div style="background-color: white; padding: 30px; border-radius: 10px; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #333; border-bottom: 2px solid #a5b4fc; padding-bottom: 10px;">
              New Message from DeepFake Website
            </h2>
            <div style="margin-top: 20px;">
              <p style="color: #666; font-size: 16px; line-height: 1.6;">
                <strong style="color: #333;">Name:</strong> ${name}
              </p>
              <p style="color: #666; font-size: 16px; line-height: 1.6;">
                <strong style="color: #333;">Email:</strong> 
                <a href="mailto:${email}" style="color: #a5b4fc; text-decoration: none;">${email}</a>
              </p>
              <div style="margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-left: 4px solid #a5b4fc; border-radius: 5px;">
                <strong style="color: #333; display: block; margin-bottom: 10px;">Message:</strong>
                <p style="color: #555; font-size: 15px; line-height: 1.8; white-space: pre-wrap;">${message}</p>
              </div>
            </div>
            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; text-align: center;">
              <p style="color: #999; font-size: 12px;">
                This message was sent from the contact form on the website
              </p>
            </div>
          </div>
        </div>
      `,
    };

    // بنرسل الإيميل
    await transporter.sendMail(mailOptions);

    console.log(" Email sent successfully:");
    console.log(" From:", name, `(${email})`);
    console.log(" Message:", message);

    // بنرجع رسالة نجاح للـ frontend
    res.status(200).json({
      success: true,
      message: "Message sent successfully!",
    });
  } catch (error) {
    console.error("Error sending email:", error);
    console.error("Error details:", error.message);
    console.error("Full error:", JSON.stringify(error, null, 2));
    
    // بنرجع تفاصيل أكثر للـ frontend عشان نعرف المشكلة
    res.status(500).json({
      success: false,
      message: "An error occurred while sending the message. Please try again",
      error: error.message, // عشان نشوف الخطأ في console الـ frontend
    });
  }
});

// endpoint بسيط عشان نتأكد إن الـ server شغال
app.get("/api/test", (req, res) => {
  res.json({ message: "Backend is running! 🎉" });
});

// بنشغل الـ server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

