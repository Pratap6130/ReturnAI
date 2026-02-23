import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

const schema = z.object({
  product_category: z.string().min(1, "Category is required"),
  product_price: z.coerce.number().positive("Price must be positive"),
  order_quantity: z.coerce.number().int().min(1, "Quantity must be at least 1"),
  user_age: z.coerce.number().int().min(1, "Age is required").max(120, "Age must be ≤ 120"),
  user_gender: z.string().min(1, "Gender is required"),
  payment_method: z.string().min(1, "Payment method is required"),
  shipping_method: z.string().min(1, "Shipping method is required"),
  discount_applied: z.coerce.number().min(0, "Discount must be ≥ 0"),
});

const CATEGORIES = [
  "Electronics", "Clothing", "Home & Kitchen", "Books", "Sports",
  "Beauty", "Toys", "Grocery", "Automotive", "Health",
];
const GENDERS = ["Male", "Female", "Other"];
const PAYMENTS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "COD"];
const SHIPPING = ["Standard", "Express", "Same Day", "Free Shipping"];

export default function PredictionForm({ onSubmit, loading }) {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({ resolver: zodResolver(schema) });

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="prediction-form">
      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="product_category">Product Category</label>
          <select id="product_category" {...register("product_category")}>
            <option value="">Select category</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          {errors.product_category && <span className="field-error">{errors.product_category.message}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="product_price">Product Price ($)</label>
          <input id="product_price" type="number" step="0.01" placeholder="29.99" {...register("product_price")} />
          {errors.product_price && <span className="field-error">{errors.product_price.message}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="order_quantity">Order Quantity</label>
          <input id="order_quantity" type="number" placeholder="1" {...register("order_quantity")} />
          {errors.order_quantity && <span className="field-error">{errors.order_quantity.message}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="user_age">Customer Age</label>
          <input id="user_age" type="number" placeholder="25" {...register("user_age")} />
          {errors.user_age && <span className="field-error">{errors.user_age.message}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="user_gender">Gender</label>
          <select id="user_gender" {...register("user_gender")}>
            <option value="">Select gender</option>
            {GENDERS.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
          {errors.user_gender && <span className="field-error">{errors.user_gender.message}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="payment_method">Payment Method</label>
          <select id="payment_method" {...register("payment_method")}>
            <option value="">Select payment</option>
            {PAYMENTS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          {errors.payment_method && <span className="field-error">{errors.payment_method.message}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="shipping_method">Shipping Method</label>
          <select id="shipping_method" {...register("shipping_method")}>
            <option value="">Select shipping</option>
            {SHIPPING.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          {errors.shipping_method && <span className="field-error">{errors.shipping_method.message}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="discount_applied">Discount Applied ($)</label>
          <input id="discount_applied" type="number" step="0.01" placeholder="0.00" {...register("discount_applied")} />
          {errors.discount_applied && <span className="field-error">{errors.discount_applied.message}</span>}
        </div>
      </div>

      <button type="submit" className="btn-primary btn-full" disabled={loading}>
        {loading ? (
          <span className="spinner-inline" />
        ) : (
          "Predict Return Risk"
        )}
      </button>
    </form>
  );
}
