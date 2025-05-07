// This is a placeholder for the Node.js entry point
// The actual functionality is in the api/index.py file

module.exports = (req, res) => {
  res.status(200).send('Redirecting to API...');
  res.redirect('/api');
}; 