package Ticketsproject2.ui;

import java.awt.EventQueue;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.table.DefaultTableModel;

import Ticketsproject2.database.DBConnection;

public class AdminDashboard extends JFrame {

	private static final long serialVersionUID = 1L;

	private JTextField textField;
	private JTextField textField_1;
	private JTable table;

	private JComboBox<String> comboBox;
	private JComboBox<String> comboBox_1;
	private JComboBox<String> comboBox_2;

	private JLabel lblNewLabel_5;
	private JLabel lblNewLabel_6;
	private JLabel lblNewLabel_7;
	private JLabel lblNewLabel_8;

	public static void main(String[] args) {
		EventQueue.invokeLater(() -> {
			try {
				AdminDashboard frame = new AdminDashboard();
				frame.setVisible(true);
			} catch (Exception e) {
				e.printStackTrace();
			}
		});
	}

	public AdminDashboard() {
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 1000, 700);
		getContentPane().setLayout(null);

		JLabel lblNewLabel = new JLabel("ADMIN DASHBOARD");
		lblNewLabel.setBounds(20, 15, 220, 30);
		getContentPane().add(lblNewLabel);

		JButton btnNewButton = new JButton("Logout");
		btnNewButton.setBounds(860, 15, 100, 30);
		getContentPane().add(btnNewButton);

		JPanel panel = new JPanel();
		panel.setLayout(null);
		panel.setBounds(20, 60, 940, 140);
		getContentPane().add(panel);

		JLabel lblNewLabel_1 = new JLabel("Total Tickets");
		lblNewLabel_1.setBounds(40, 15, 100, 20);
		panel.add(lblNewLabel_1);

		JLabel lblNewLabel_2 = new JLabel("Open Tickets");
		lblNewLabel_2.setBounds(260, 15, 100, 20);
		panel.add(lblNewLabel_2);

		JLabel lblNewLabel_3 = new JLabel("Closed Tickets");
		lblNewLabel_3.setBounds(480, 15, 110, 20);
		panel.add(lblNewLabel_3);

		JLabel lblNewLabel_4 = new JLabel("High Priority");
		lblNewLabel_4.setBounds(720, 15, 100, 20);
		panel.add(lblNewLabel_4);

		lblNewLabel_5 = new JLabel("0");
		lblNewLabel_5.setBounds(75, 45, 40, 20);
		panel.add(lblNewLabel_5);

		lblNewLabel_6 = new JLabel("0");
		lblNewLabel_6.setBounds(300, 45, 40, 20);
		panel.add(lblNewLabel_6);

		lblNewLabel_7 = new JLabel("0");
		lblNewLabel_7.setBounds(525, 45, 40, 20);
		panel.add(lblNewLabel_7);

		lblNewLabel_8 = new JLabel("0");
		lblNewLabel_8.setBounds(760, 45, 40, 20);
		panel.add(lblNewLabel_8);

		JLabel lblNewLabel_10 = new JLabel("Selected ID:");
		lblNewLabel_10.setBounds(40, 90, 90, 20);
		panel.add(lblNewLabel_10);

		textField_1 = new JTextField();
		textField_1.setBounds(135, 87, 120, 28);
		textField_1.setColumns(10);
		textField_1.setEditable(false);
		panel.add(textField_1);

		JLabel lblNewLabel_11 = new JLabel("Status:");
		lblNewLabel_11.setBounds(360, 90, 60, 20);
		panel.add(lblNewLabel_11);

		comboBox_1 = new JComboBox<>();
		comboBox_1.setModel(new DefaultComboBoxModel<>(new String[] {
				"Open", "In Progress", "Closed"
		}));
		comboBox_1.setBounds(420, 87, 150, 28);
		panel.add(comboBox_1);

		JLabel lblNewLabel_12 = new JLabel("Assign Agent:");
		lblNewLabel_12.setBounds(650, 90, 100, 20);
		panel.add(lblNewLabel_12);

		comboBox_2 = new JComboBox<>();
		comboBox_2.setModel(new DefaultComboBoxModel<>(new String[] {
				"agent1", "agent2"
		}));
		comboBox_2.setBounds(760, 87, 130, 28);
		panel.add(comboBox_2);

		textField = new JTextField();
		textField.setBounds(20, 220, 180, 30);
		textField.setColumns(10);
		getContentPane().add(textField);

		JLabel lblNewLabel_9 = new JLabel("priority:");
		lblNewLabel_9.setBounds(220, 225, 60, 20);
		getContentPane().add(lblNewLabel_9);

		comboBox = new JComboBox<>();
		comboBox.setModel(new DefaultComboBoxModel<>(new String[] {
				"All", "Low", "Medium", "High", "Critical"
		}));
		comboBox.setBounds(285, 220, 130, 30);
		getContentPane().add(comboBox);

		JButton btnNewButton_1 = new JButton("Search");
		btnNewButton_1.setBounds(440, 220, 100, 30);
		getContentPane().add(btnNewButton_1);

		JButton btnNewButton_2 = new JButton("Refresh");
		btnNewButton_2.setBounds(560, 220, 100, 30);
		getContentPane().add(btnNewButton_2);

		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setBounds(20, 270, 940, 300);
		getContentPane().add(scrollPane);

		table = new JTable();
		table.setModel(new DefaultTableModel(
			new Object[][] {
			},
			new String[] {
				"ID", "Customer", "Issue", "Priority", "Assigned Agent", "Status"
			}
		) {
			private static final long serialVersionUID = 1L;

			@Override
			public boolean isCellEditable(int row, int column) {
				return false;
			}
		});
		scrollPane.setViewportView(table);

		JButton btnNewButton_3 = new JButton("Update");
		btnNewButton_3.setBounds(20, 590, 120, 35);
		getContentPane().add(btnNewButton_3);

		JButton btnNewButton_4 = new JButton("Delete");
		btnNewButton_4.setBounds(160, 590, 120, 35);
		getContentPane().add(btnNewButton_4);

		btnNewButton.addActionListener(e -> dispose());
		btnNewButton_1.addActionListener(e -> searchTicketById());
		btnNewButton_2.addActionListener(e -> refreshDashboard());
		btnNewButton_3.addActionListener(e -> updateTicket());
		btnNewButton_4.addActionListener(e -> deleteTicket());

		table.getSelectionModel().addListSelectionListener(e -> {
			if (!e.getValueIsAdjusting()) {
				fillFieldsFromTable();
			}
		});

		loadAllTickets();
		loadStats();
	}

	private void loadAllTickets() {
		DefaultTableModel model = (DefaultTableModel) table.getModel();
		model.setRowCount(0);

		String selectedPriority = comboBox.getSelectedItem().toString();
		String sql = "SELECT id, customerName, issue, priority, assignedAgent, status FROM tickets";
		boolean hasPriorityFilter = !"All".equals(selectedPriority);

		if (hasPriorityFilter) {
			sql += " WHERE priority = ?";
		}

		sql += " ORDER BY id DESC";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			if (hasPriorityFilter) {
				pst.setString(1, selectedPriority);
			}

			try (ResultSet rs = pst.executeQuery()) {
				while (rs.next()) {
					model.addRow(new Object[] {
						rs.getInt("id"),
						rs.getString("customerName"),
						rs.getString("issue"),
						rs.getString("priority"),
						rs.getString("assignedAgent"),
						rs.getString("status")
					});
				}
			}
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Error loading tickets: " + e.getMessage());
		}
	}

	private void searchTicketById() {
		String idText = textField.getText().trim();

		if (idText.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Enter ticket ID");
			return;
		}

		DefaultTableModel model = (DefaultTableModel) table.getModel();
		model.setRowCount(0);

		String sql = "SELECT id, customerName, issue, priority, assignedAgent, status FROM tickets WHERE id = ?";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setInt(1, Integer.parseInt(idText));

			try (ResultSet rs = pst.executeQuery()) {
				if (rs.next()) {
					model.addRow(new Object[] {
						rs.getInt("id"),
						rs.getString("customerName"),
						rs.getString("issue"),
						rs.getString("priority"),
						rs.getString("assignedAgent"),
						rs.getString("status")
					});

					textField_1.setText(String.valueOf(rs.getInt("id")));
					comboBox_1.setSelectedItem(rs.getString("status"));
					comboBox_2.setSelectedItem(rs.getString("assignedAgent"));
				} else {
					JOptionPane.showMessageDialog(this, "Ticket not found");
				}
			}
		} catch (NumberFormatException e) {
			JOptionPane.showMessageDialog(this, "Ticket ID must be a number");
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Search error: " + e.getMessage());
		}
	}

	private void fillFieldsFromTable() {
		int row = table.getSelectedRow();

		if (row >= 0) {
			textField_1.setText(table.getValueAt(row, 0).toString());
			comboBox_2.setSelectedItem(table.getValueAt(row, 4).toString());
			comboBox_1.setSelectedItem(table.getValueAt(row, 5).toString());
		}
	}

	private void updateTicket() {
		String selectedId = textField_1.getText().trim();

		if (selectedId.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Select a ticket first");
			return;
		}

		String sql = "UPDATE tickets SET status = ?, assignedAgent = ? WHERE id = ?";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setString(1, comboBox_1.getSelectedItem().toString());
			pst.setString(2, comboBox_2.getSelectedItem().toString());
			pst.setInt(3, Integer.parseInt(selectedId));

			pst.executeUpdate();

			JOptionPane.showMessageDialog(this, "Ticket updated successfully");
			loadAllTickets();
			loadStats();
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Update error: " + e.getMessage());
		}
	}

	private void deleteTicket() {
		String selectedId = textField_1.getText().trim();

		if (selectedId.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Select a ticket first");
			return;
		}

		String sql = "DELETE FROM tickets WHERE id = ?";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setInt(1, Integer.parseInt(selectedId));
			pst.executeUpdate();

			JOptionPane.showMessageDialog(this, "Ticket deleted successfully");
			refreshDashboard();
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Delete error: " + e.getMessage());
		}
	}

	private void refreshDashboard() {
		textField.setText("");
		textField_1.setText("");
		comboBox.setSelectedIndex(0);
		comboBox_1.setSelectedItem("Open");
		comboBox_2.setSelectedItem("agent1");
		loadAllTickets();
		loadStats();
	}

	private void loadStats() {
		lblNewLabel_5.setText(String.valueOf(getCount("SELECT COUNT(*) FROM tickets")));
		lblNewLabel_6.setText(String.valueOf(getCount("SELECT COUNT(*) FROM tickets WHERE status = 'Open'")));
		lblNewLabel_7.setText(String.valueOf(getCount("SELECT COUNT(*) FROM tickets WHERE status = 'Closed'")));
		lblNewLabel_8.setText(String.valueOf(getCount("SELECT COUNT(*) FROM tickets WHERE priority = 'High'")));
	}

	private int getCount(String sql) {
		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql);
			 ResultSet rs = pst.executeQuery()) {

			if (rs.next()) {
				return rs.getInt(1);
			}
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Stats error: " + e.getMessage());
		}
		return 0;
	}
}
